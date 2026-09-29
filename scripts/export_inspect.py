"""Convert Real-Claw-Bench results to Inspect AI .eval format.

Usage:
    python3 scripts/export_inspect.py \
        --runs runs/full \
        --traces traces/full \
        --cases data/benchmark/cases \
        --out inspect_logs/

Each model gets its own .eval file: inspect_logs/<model_label>.eval
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_cases_meta(cases_jsonl: Path) -> dict[str, dict]:
    """Load case metadata (id, category, etc.) from cases.jsonl."""
    meta = {}
    for line in cases_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        meta[c.get("id") or c.get("case_id")] = c
    return meta


def read_task_md(cases_root: Path, case_id: str) -> str:
    task_path = cases_root / case_id / "task.md"
    if task_path.exists():
        return task_path.read_text(encoding="utf-8")
    return f"Task {case_id}"


def _content_to_text(content) -> str:
    """Flatten an OpenClaw message `content` (str or list of typed parts) to text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for part in content:
        if not isinstance(part, dict):
            out.append(str(part))
            continue
        t = part.get("type")
        if t == "text":
            out.append(part.get("text", ""))
        elif t == "thinking":
            think = part.get("thinking", "")
            if think:
                out.append(f"[reasoning]\n{think}")
        elif t == "image":
            out.append("[image omitted]")
        # toolCall parts are handled separately (turned into ToolCall objects)
    return "\n\n".join(s for s in out if s)


def build_transcript(session, task_text):
    """Convert an OpenClaw session jsonl (list of records) into Inspect chat
    messages, preserving each step: user turns, assistant reasoning + tool calls,
    and tool results. This is what makes the trace genuinely followable."""
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageTool,
        ChatMessageUser,
    )
    from inspect_ai.tool import ToolCall

    if not session:
        return None

    messages = []
    for rec in session:
        if not isinstance(rec, dict):
            continue
        rtype = rec.get("type")
        if rtype == "compaction":
            messages.append(
                ChatMessageSystem(content="[context compaction: older turns summarized]")
            )
            continue
        if rtype != "message":
            continue
        msg = rec.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            messages.append(ChatMessageUser(content=_content_to_text(content) or ""))
        elif role == "assistant":
            tool_calls = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "toolCall":
                        args = part.get("arguments")
                        if not isinstance(args, dict):
                            args = {"_raw": part.get("partialJson") or args}
                        tool_calls.append(
                            ToolCall(
                                id=str(part.get("id") or ""),
                                function=str(part.get("name") or "unknown"),
                                arguments=args,
                            )
                        )
            messages.append(
                ChatMessageAssistant(
                    content=_content_to_text(content) or "",
                    tool_calls=tool_calls or None,
                )
            )
        elif role == "toolResult":
            text = _content_to_text(content)
            is_err = bool(msg.get("isError") or msg.get("error"))
            messages.append(
                ChatMessageTool(
                    content=text,
                    tool_call_id=str(msg.get("toolCallId") or ""),
                    function=str(msg.get("toolName") or ""),
                    error=({"type": "unknown", "message": text[:500]} if is_err else None),
                )
            )

    # Drop a leading user turn equal to nothing; ensure the task is the first thing shown.
    if messages and not any(isinstance(m, ChatMessageUser) for m in messages[:1]):
        messages.insert(0, ChatMessageUser(content=task_text))
    return messages or None


def convert_model(
    model_label: str,
    runs_jsonl: Path,
    traces_dir: Path,
    cases_root: Path,
    cases_meta: dict,
    out_path: Path,
) -> None:
    from inspect_ai.log import (
        EvalConfig,
        EvalDataset,
        EvalLog,
        EvalMetric,
        EvalPlan,
        EvalResults,
        EvalSample,
        EvalScore,
        EvalSpec,
        EvalStats,
    )
    from inspect_ai.log._file import write_eval_log
    from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
    from inspect_ai.scorer import Score

    rows = [json.loads(l) for l in runs_jsonl.read_text().splitlines() if l.strip()]

    samples = []
    n_pass = 0

    for r in rows:
        case_id = r["case_id"]
        task_text = read_task_md(cases_root, case_id)
        case_meta = cases_meta.get(case_id, {})

        # Verifier checks → score breakdown
        verifier = r.get("verifier") or {}
        checks = verifier.get("checks") or []
        n_checks = len(checks)
        n_check_pass = sum(1 for c in checks if c["ok"])
        passed = r.get("ok", False)
        if passed:
            n_pass += 1

        # Score value: fraction of checks passing, or 0/1 if no checks
        if n_checks > 0:
            score_value = n_check_pass / n_checks
        else:
            score_value = 1.0 if passed else 0.0

        # Build explanation from checks
        check_lines = []
        for c in checks:
            mark = "✓" if c["ok"] else "✗"
            detail = str(c.get("detail", ""))[:120]
            check_lines.append(f"{mark} {c['name']}: {detail}")
        explanation = "\n".join(check_lines) if check_lines else (r.get("error") or "")

        # Load trace for messages + agent state. Build the FULL step-by-step
        # transcript from the captured session jsonl when available, so the
        # strategy/plan, each tool call + result, and the failure mode are visible.
        messages = [ChatMessageUser(content=task_text)]
        elapsed = r.get("elapsed_s")
        aborted = False
        stop_reason = None
        tool_summary: dict = {}
        exec_trace: dict = {}
        context_mgmt: dict = {}
        usage: dict = {}
        final_text = ""

        # Tolerate both the fixed layout (traces/hpc/<model>/<case>.json) and the
        # older double-nested one (traces/hpc/<model>/<model>/<case>.json).
        trace_candidates = [
            traces_dir / f"{case_id}.json",
            traces_dir / traces_dir.name / f"{case_id}.json",
        ]
        trace_file = next((p for p in trace_candidates if p.exists()), None)
        if trace_file is not None:
            try:
                trace = json.loads(trace_file.read_text())
                meta = (trace.get("output_json") or {}).get("meta") or {}
                aborted = bool(meta.get("aborted"))
                stop_reason = meta.get("stopReason")
                tool_summary = meta.get("toolSummary") or {}
                exec_trace = meta.get("executionTrace") or {}
                context_mgmt = meta.get("contextManagement") or {}
                usage = (meta.get("agentMeta") or {}).get("usage") or {}
                payloads = (trace.get("output_json") or {}).get("payloads") or []
                final_text = payloads[0].get("text", "") if payloads else ""
                transcript = build_transcript(trace.get("session"), task_text)
                if transcript:
                    messages = transcript
                elif final_text:
                    messages.append(ChatMessageAssistant(content=final_text))
            except Exception:
                pass

        # Score metadata: surfaces strategy/state/failure-mode for quick triage.
        score_meta: dict = {
            "case_id": case_id,
            "elapsed_s": elapsed,
            "aborted": aborted,
            "stop_reason": stop_reason,
            "tool_calls": tool_summary.get("calls"),
            "tools_used": tool_summary.get("tools"),
            "tool_failures": tool_summary.get("failures"),
            "fallback_used": exec_trace.get("fallbackUsed"),
            "compactions": context_mgmt.get("lastTurnCompactions"),
            "tokens": usage,
            "n_checks_pass": n_check_pass,
            "n_checks_total": n_checks,
            "returncode": r.get("returncode"),
        }
        if case_meta:
            score_meta["category"] = case_meta.get("category", "")

        sample = EvalSample(
            id=case_id,
            epoch=1,
            input=task_text,
            target="PASS",
            messages=messages,
            scores={
                "verifier": Score(
                    value=score_value,
                    answer="PASS" if passed else "FAIL",
                    explanation=explanation,
                    metadata=score_meta,
                )
            },
            metadata={
                "case_id": case_id,
                "category": case_meta.get("category", ""),
                "elapsed_s": elapsed,
                "aborted": aborted,
                "stop_reason": stop_reason,
                "tool_summary": tool_summary,
                "execution_trace": exec_trace,
                "context_management": context_mgmt,
                "tokens": usage,
                "final_text": final_text[:2000],
                "checks": checks,
            },
        )
        samples.append(sample)

    # Aggregate score
    total = len(rows)
    pass_rate = n_pass / total if total > 0 else 0.0

    now = datetime.now(timezone.utc).isoformat()

    log = EvalLog(
        version=2,
        status="success",
        eval=EvalSpec(
            created=now,
            task="real-claw-bench",
            task_id=f"real-claw-bench/{model_label}",
            task_display_name="Real-Claw-Bench",
            dataset=EvalDataset(
                name="real-claw-bench",
                location=str(cases_root),
            ),
            model=model_label,
            config=EvalConfig(),
            metadata={"model_label": model_label},
        ),
        plan=EvalPlan(name="agent"),
        results=EvalResults(
            total_samples=total,
            completed_samples=total,
            scores=[
                EvalScore(
                    name="verifier",
                    scorer="verifier",
                    params={},
                    metrics={
                        "accuracy": EvalMetric(name="accuracy", value=pass_rate),
                        "pass_count": EvalMetric(name="pass_count", value=float(n_pass)),
                    },
                )
            ],
        ),
        stats=EvalStats(started_at=now, completed_at=now),
        samples=samples,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, str(out_path))
    print(f"  {model_label}: {n_pass}/{total} pass ({pass_rate:.1%}) → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark results to Inspect AI .eval format")
    parser.add_argument("--runs", type=Path, default=Path("runs/full"), help="Directory with <model>.jsonl files")
    parser.add_argument("--traces", type=Path, default=Path("traces/full"), help="Directory with traces/<model>/<case>.json")
    parser.add_argument("--cases", type=Path, default=Path("data/benchmark/cases"), help="Benchmark cases root")
    parser.add_argument("--cases-jsonl", type=Path, default=Path("data/benchmark/cases.jsonl"), help="Cases JSONL index")
    parser.add_argument("--out", type=Path, default=Path("inspect_logs"), help="Output directory for .eval files")
    parser.add_argument("--models", default="", help="Comma-separated model labels to export (default: all)")
    args = parser.parse_args()

    from inspect_ai.log import EvalMetric

    cases_meta = load_cases_meta(args.cases_jsonl)
    print(f"Loaded {len(cases_meta)} case metadata entries")

    filter_models = set(args.models.split(",")) if args.models else set()

    jsonl_files = sorted(args.runs.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {args.runs}")
        return

    print(f"Exporting {len(jsonl_files)} models to {args.out}/")
    for jsonl in jsonl_files:
        label = jsonl.stem
        if filter_models and label not in filter_models:
            continue
        traces_model_dir = args.traces / label
        out_file = args.out / f"{label}.eval"
        try:
            convert_model(
                model_label=label,
                runs_jsonl=jsonl,
                traces_dir=traces_model_dir,
                cases_root=args.cases,
                cases_meta=cases_meta,
                out_path=out_file,
            )
        except Exception as e:
            print(f"  {label}: ERROR — {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
