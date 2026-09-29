"""Convert a Terminal-Bench (Harbor) run made with the OpenClaw agent into
Inspect AI .eval format -- the TB analogue of scripts/export_inspect.py.

Data sources (all produced by a normal TB run + our proxy capture):
  * <run>/results.json        per-task verdict: task_id, instruction, is_resolved,
                              failure_mode, parser_results (per-test pass/fail), tokens, timings.
  * <run>/run_metadata.json   model_name, dataset name/version, agent_kwargs.
  * <capture>.jsonl           the proxy --capture-file: one record per LLM call with the
                              full OpenAI-format `request` (messages, tools) + streamed `response`.

The capture has no task tag, so each record is mapped to a task by matching its first
`user` message text against that task's `instruction`. Within a task, the call with the
most messages is the LAST turn; its `messages` already contain every prior assistant/tool
turn, and the final assistant turn is reconstructed by aggregating that call's SSE stream.

Usage (run under the RCB .venv that has inspect_ai):
    python3 scripts/export_inspect_tb.py \
        --run runs/tbench/smoke7/2026-06-23__00-25-46 \
        --capture traces/tbench/_raw_calls_smoke7.jsonl \
        --out inspect_logs/tbench
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _norm(s: str) -> str:
    """Collapse whitespace for robust instruction<->message matching."""
    return re.sub(r"\s+", " ", (s or "")).strip()


def _user_text(content) -> str:
    """Flatten an OpenAI user `content` (str or list of {type,text}) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _first_user_text(messages) -> str:
    for m in messages:
        if m.get("role") == "user":
            return _user_text(m.get("content"))
    return ""


def aggregate_stream(sse: str) -> tuple[str, list[dict]]:
    """Aggregate an OpenAI streamed (SSE) response into (text, tool_calls).

    tool_calls is a list of {id, name, arguments(str)} reconstructed by `index`.
    """
    text_parts: list[str] = []
    tool_acc: dict[int, dict] = {}
    for line in sse.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        try:
            chunk = json.loads(body)
        except Exception:
            continue
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                acc = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["name"] = fn["name"]
                if fn.get("arguments"):
                    acc["arguments"] += fn["arguments"]
    tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    return "".join(text_parts), tool_calls


def _make_tool_call(call_id: str, name: str, arguments):
    from inspect_ai.tool import ToolCall

    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except Exception:
            args = {"_raw": arguments}
    elif isinstance(arguments, dict):
        args = arguments
    else:
        args = {"_raw": str(arguments)}
    return ToolCall(id=str(call_id or ""), function=str(name or "unknown"), arguments=args)


def build_transcript(messages, final_text, final_tool_calls, task_text):
    """Convert captured OpenAI-format messages (+ the reconstructed final assistant
    turn) into Inspect chat messages."""
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageTool,
        ChatMessageUser,
    )

    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append(ChatMessageSystem(content=m.get("content") or ""))
        elif role == "user":
            out.append(ChatMessageUser(content=_user_text(m.get("content")) or ""))
        elif role == "assistant":
            tcs = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tcs.append(_make_tool_call(tc.get("id"), fn.get("name"), fn.get("arguments")))
            out.append(
                ChatMessageAssistant(content=m.get("content") or "", tool_calls=tcs or None)
            )
        elif role == "tool":
            out.append(
                ChatMessageTool(
                    content=m.get("content") or "",
                    tool_call_id=str(m.get("tool_call_id") or ""),
                )
            )

    # Append the final assistant turn reconstructed from the last call's stream.
    if final_text or final_tool_calls:
        tcs = [_make_tool_call(t["id"], t["name"], t["arguments"]) for t in final_tool_calls]
        out.append(ChatMessageAssistant(content=final_text or "", tool_calls=tcs or None))

    if not out:
        out = [ChatMessageUser(content=task_text)]
    return out


def index_capture(capture_path: Path):
    """Return list of records each annotated with its first-user-text and message count."""
    recs = []
    for line in capture_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        msgs = (d.get("request") or {}).get("messages") or []
        recs.append({"rec": d, "ukey": _norm(_first_user_text(msgs)), "nmsg": len(msgs)})
    return recs


def match_task_record(recs, instruction: str):
    """Pick the LAST-turn capture record (most messages) whose first user message
    matches this task's instruction (containment either way, normalized)."""
    key = _norm(instruction)
    if not key:
        return None
    cand = []
    for r in recs:
        u = r["ukey"]
        if not u:
            continue
        if key in u or u in key or key[:80] in u:
            cand.append(r)
    if not cand:
        return None
    return max(cand, key=lambda r: r["nmsg"])


def convert_run(run_dir: Path, capture_path: Path, out_dir: Path) -> None:
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
    from inspect_ai.scorer import Score

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())

    raw_model = meta.get("model_name") or "unknown"
    model_label = raw_model.split("/")[-1]  # azure-openai/gpt-4o_inference -> gpt-4o_inference

    recs = index_capture(capture_path) if capture_path and capture_path.exists() else []

    samples = []
    n_pass = 0
    rows = results.get("results") or []

    for r in rows:
        task_id = r.get("task_id")
        instruction = r.get("instruction") or task_id or ""
        passed = bool(r.get("is_resolved"))
        if passed:
            n_pass += 1

        parser_results = r.get("parser_results") or {}
        n_checks = len(parser_results)
        n_check_pass = sum(1 for v in parser_results.values() if str(v).lower() == "passed")
        score_value = (n_check_pass / n_checks) if n_checks else (1.0 if passed else 0.0)

        check_lines = [
            f"{'✓' if str(v).lower() == 'passed' else '✗'} {k}: {v}"
            for k, v in parser_results.items()
        ]
        explanation = "\n".join(check_lines) if check_lines else (r.get("failure_mode") or "")

        # transcript from the matched capture record
        match = match_task_record(recs, instruction)
        final_text, final_tcs = "", []
        if match is not None:
            msgs = (match["rec"].get("request") or {}).get("messages") or []
            final_text, final_tcs = aggregate_stream(match["rec"].get("response") or "")
            messages = build_transcript(msgs, final_text, final_tcs, instruction)
        else:
            from inspect_ai.model import ChatMessageUser

            messages = [ChatMessageUser(content=instruction)]

        def _elapsed(a, b):
            try:
                return (
                    datetime.fromisoformat(r[b]) - datetime.fromisoformat(r[a])
                ).total_seconds()
            except Exception:
                return None

        score_meta = {
            "task_id": task_id,
            "is_resolved": passed,
            "failure_mode": r.get("failure_mode"),
            "n_checks_pass": n_check_pass,
            "n_checks_total": n_checks,
            "agent_elapsed_s": _elapsed("agent_started_at", "agent_ended_at"),
            "test_elapsed_s": _elapsed("test_started_at", "test_ended_at"),
            "input_tokens": r.get("total_input_tokens"),
            "output_tokens": r.get("total_output_tokens"),
            "n_llm_calls_matched": match["nmsg"] if match else 0,
        }

        samples.append(
            EvalSample(
                id=task_id,
                epoch=1,
                input=instruction,
                target="PASS",
                messages=messages,
                scores={
                    "tb_verifier": Score(
                        value=score_value,
                        answer="PASS" if passed else "FAIL",
                        explanation=explanation,
                        metadata=score_meta,
                    )
                },
                metadata={
                    "task_id": task_id,
                    "failure_mode": r.get("failure_mode"),
                    "parser_results": parser_results,
                    "final_text": (final_text or "")[:2000],
                    "trial_name": r.get("trial_name"),
                    "recording_path": r.get("recording_path"),
                },
            )
        )

    total = len(rows)
    pass_rate = n_pass / total if total else 0.0
    now = datetime.now(timezone.utc).isoformat()

    log = EvalLog(
        version=2,
        status="success",
        eval=EvalSpec(
            created=now,
            task="terminal-bench",
            task_id=f"terminal-bench/{model_label}",
            task_display_name="Terminal-Bench (OpenClaw)",
            dataset=EvalDataset(
                name=f"{meta.get('dataset_name')}=={meta.get('dataset_version')}",
                location=str(run_dir),
            ),
            model=model_label,
            config=EvalConfig(),
            metadata={
                "model_name": raw_model,
                "agent_name": meta.get("agent_name"),
                "agent_kwargs": meta.get("agent_kwargs"),
                "run_id": meta.get("run_id"),
            },
        ),
        plan=EvalPlan(name="openclaw"),
        results=EvalResults(
            total_samples=total,
            completed_samples=total,
            scores=[
                EvalScore(
                    name="tb_verifier",
                    scorer="tb_verifier",
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

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_label}.eval"
    write_eval_log(log, str(out_path))
    matched = sum(1 for s in samples if s.scores["tb_verifier"].metadata["n_llm_calls_matched"])
    print(
        f"  {model_label}: {n_pass}/{total} pass ({pass_rate:.1%}), "
        f"transcripts matched {matched}/{total} → {out_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export a Terminal-Bench OpenClaw run to Inspect .eval")
    p.add_argument("--run", type=Path, required=True, help="A TB run dir (the one with results.json)")
    p.add_argument("--capture", type=Path, required=True, help="proxy --capture-file jsonl for this run")
    p.add_argument("--out", type=Path, default=Path("inspect_logs/tbench"), help="output dir for .eval")
    args = p.parse_args()

    if not (args.run / "results.json").is_file():
        raise SystemExit(f"no results.json under {args.run}")
    convert_run(args.run, args.capture, args.out)


if __name__ == "__main__":
    main()
