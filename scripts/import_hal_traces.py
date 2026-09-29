#!/usr/bin/env python3
"""Import HAL (Holistic Agent Leaderboard) agent traces into Inspect AI .eval logs.

HAL publishes full agent traces on HuggingFace (agent-evals/hal_traces) as
encrypted zips. Each zip holds one JSON {salt, encrypted_data}; the payload is a
Fernet ciphertext (PBKDF2-HMAC-SHA256, 480k iters, password 'hal1234').

Decrypted payload (full / non-slim traces) has:
  config              run metadata (agent_name, benchmark_name, run_id, ...)
  results             accuracy + successful_tasks / failed_tasks + latencies
  raw_eval_results    per-task grading detail (shape varies per benchmark)
  raw_logging_results list of W&B Weave litellm.completion calls, each with
                      inputs (request: model + messages) and output (response)
  total_usage / total_cost / git_info

Model calls are joined to tasks by `weave_task_id`, which matches the task ids
in results.successful_tasks / failed_tasks.

"slim" traces have no raw_logging_results (labels only, no transcript) and are
skipped unless --allow-slim is given.

Usage:
  python scripts/import_hal_traces.py --zip a.zip [--zip b.zip ...] --out inspect_logs/hal
  python scripts/import_hal_traces.py --dir traces/hal_zips --out inspect_logs/hal
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PASSWORD = b"hal1234"


def free_gb(path: Path) -> float:
    """Free space (GiB) on the filesystem holding `path` (nearest existing parent)."""
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / (1024 ** 3)


def decrypt_zip(zip_path: Path, password: bytes = DEFAULT_PASSWORD) -> dict:
    """Decrypt a HAL trace zip to its JSON payload."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    with zipfile.ZipFile(zip_path) as zf:
        enc = json.loads(zf.read(zf.namelist()[0]))
    salt = base64.b64decode(enc["salt"].encode())
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    key = base64.urlsafe_b64encode(kdf.derive(password))
    plaintext = Fernet(key).decrypt(base64.b64decode(enc["encrypted_data"].encode()))
    return json.loads(plaintext.decode())


def _content_to_text(content) -> str:
    """Flatten OpenAI-style message content (str or list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def _output_text(output) -> str:
    """Extract assistant text from a litellm/OpenAI completion response."""
    if not output:
        return ""
    if isinstance(output, str):
        return output
    choices = output.get("choices") if isinstance(output, dict) else None
    if choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return _content_to_text(msg.get("content"))
    return ""


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _eval_stem(zip_name: str) -> str:
    """Derive a stable .eval basename from a HAL zip filename.

    The zip name already encodes benchmark + agent + model + run_id, e.g.
    scicode_scicode_zero_shot_agent_..._1745456160_UPLOAD.zip
    """
    stem = zip_name
    for suf in (".zip", "_UPLOAD"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return _sanitize(stem)


HF_REPO = "agent-evals/hal_traces"

# Per-task binary verdict fields seen across HAL benchmark schemas.
_VERDICT_FIELDS = ("success_rate", "score", "correct", "is_correct",
                   "passed", "resolved", "reward", "accuracy")


def _verdict_value(entry) -> float | None:
    """Extract a numeric pass/fail signal (0..1) from a per-task result entry."""
    if isinstance(entry, bool):
        return 1.0 if entry else 0.0
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for f in _VERDICT_FIELDS:
            if f in entry:
                v = entry[f]
                if isinstance(v, bool):
                    return 1.0 if v else 0.0
                if isinstance(v, (int, float)):
                    return float(v)
    return None


def resolve_labels(data: dict) -> tuple[dict[str, float], str]:
    """Return {task_id: verdict(0..1)} and the strategy name used.

    Handles the differing HAL result schemas:
      A) results.successful_tasks / failed_tasks (GAIA, scicode)
      B) raw_eval_results.eval_result[tid].success_rate (scienceagentbench)
      C) raw_eval_results[tid] = {score, ...} or numeric (GAIA slim, others)
    """
    res = data.get("results", {}) or {}
    passed = [str(t) for t in res.get("successful_tasks", []) or []]
    failed = [str(t) for t in res.get("failed_tasks", []) or []]
    if passed or failed:
        labels = {t: 1.0 for t in passed}
        labels.update({t: 0.0 for t in failed})
        return labels, "results.successful/failed_tasks"

    rer = data.get("raw_eval_results")
    # D) raw_eval_results as a positional list of per-task correctness scores
    #    (colbench): list index == integer task id; mean == results.average_correctness.
    #    colbench's headline metric is accuracy == fraction with full correctness (1.0),
    #    so treat score 1.0 as pass and any partial/zero score as fail.
    #    The list spans all 1000 task slots but only the tasks that actually ran carry
    #    a latency entry (== the tasks that have a trace); restrict labels to those so
    #    we never emit label-only samples with an empty transcript.
    if isinstance(rer, list) and rer:
        ran = {str(k) for k in (res.get("latencies") or {})}
        labels = {}
        for tid, entry in enumerate(rer):
            if ran and str(tid) not in ran:
                continue
            v = _verdict_value(entry)
            if v is not None:
                labels[str(tid)] = 1.0 if v >= 0.999 else 0.0
        if labels:
            return labels, "raw_eval_results[list]"
    if isinstance(rer, dict) and rer:
        # B) nested eval_result dict keyed by task id
        for sub_key in ("eval_result", "eval_results", "details", "scores"):
            sub = rer.get(sub_key)
            if isinstance(sub, dict) and sub:
                labels = {}
                for tid, entry in sub.items():
                    v = _verdict_value(entry)
                    if v is not None:
                        labels[str(tid)] = 1.0 if v >= 0.5 else 0.0
                if labels:
                    return labels, f"raw_eval_results.{sub_key}"
        # C) raw_eval_results keyed directly by task id
        labels = {}
        for tid, entry in rer.items():
            v = _verdict_value(entry)
            if v is not None:
                labels[str(tid)] = 1.0 if v >= 0.5 else 0.0
        if labels:
            return labels, "raw_eval_results[task]"
    return {}, "none"


def list_hf_zips(repo: str, prefixes: list[str]) -> list[str]:
    """List trace zip filenames in the HF dataset, optionally filtered by prefix."""
    import urllib.request

    url = f"https://huggingface.co/api/datasets/{repo}"
    req = urllib.request.Request(url, headers={"User-Agent": "hal-import"})
    with urllib.request.urlopen(req, timeout=60) as r:
        meta = json.loads(r.read().decode())
    names = [s["rfilename"] for s in meta.get("siblings", [])
             if s["rfilename"].endswith(".zip")]
    if prefixes:
        names = [n for n in names if any(n.startswith(p) for p in prefixes)]
    return sorted(names)


def hf_file_size(repo: str, filename: str) -> int:
    """Return the remote size (bytes) of one HF dataset file via a HEAD request.

    Lets us skip pathologically large traces before downloading them. Returns 0
    if the size can't be determined (caller then falls back to a post-download
    check on the actual file size).
    """
    import urllib.request

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "hal-import"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001 - size probe is best-effort
        return 0


def download_hf_file(repo: str, filename: str, dest: Path, retries: int = 4) -> None:
    """Stream one file from the HF dataset to dest, verifying completeness.

    Large LFS files over a flaky link can truncate silently (the socket ends
    early), yielding a corrupt zip. We check bytes-written against Content-Length
    and retry a few times before giving up.
    """
    import time
    import urllib.request

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hal-import"})
            with urllib.request.urlopen(req, timeout=120) as r:
                expected = int(r.headers.get("Content-Length") or 0)
                written = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
            if expected and written != expected:
                raise IOError(f"truncated: got {written} of {expected} bytes")
            return
        except Exception as e:  # noqa: BLE001 - retry transient network failures
            last_err = e
            dest.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(3 * attempt)
    raise IOError(f"download failed after {retries} attempts: {last_err}")


def process_zip(zip_path: Path, out_dir: Path, min_output_rate: float,
                allow_slim: bool, stem: str):
    """Decrypt one zip and write its .eval. Returns (status, note).

    status in {"ok", "skip", "err"}. On skip/err no .eval is written.
    """
    from inspect_ai.log._file import write_eval_log
    try:
        data = decrypt_zip(zip_path)
        log, note = build_eval_log(data, min_output_rate, allow_slim)
    except Exception as e:  # noqa: BLE001 - report and continue per-file
        return "err", str(e)
    if log is None:
        return "skip", note
    write_eval_log(log, str(out_dir / f"{stem}.eval"))
    return "ok", note


def build_eval_log(data: dict, min_output_rate: float, allow_slim: bool):
    """Convert one decrypted HAL trace dict into (EvalLog, note), or (None, note) to skip."""
    from inspect_ai.log import (
        EvalConfig, EvalDataset, EvalLog, EvalMetric, EvalPlan,
        EvalResults, EvalSample, EvalScore, EvalSpec, EvalStats,
    )
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
    from inspect_ai.scorer import Score

    cfg = data.get("config", {})
    res = data.get("results", {})
    calls = data.get("raw_logging_results") or []
    agent_name = cfg.get("agent_name", "unknown_agent")
    benchmark = cfg.get("benchmark_name", "unknown")
    run_id = cfg.get("run_id", "")
    latencies = res.get("latencies", {}) or {}

    if not calls:
        if not allow_slim:
            return None, "slim (no raw_logging_results) — skipped"
        health = "slim: labels only, no transcript"
    else:
        with_out = sum(1 for c in calls if c.get("output"))
        rate = with_out / len(calls) if calls else 0.0
        if rate < min_output_rate:
            return None, f"degenerate run (output_rate={rate:.0%} < {min_output_rate:.0%}) — skipped"
        health = f"output_rate={rate:.0%}, {len(calls)} calls"

    # Labels: schema varies per benchmark; resolve_labels handles the variants.
    labels, label_src = resolve_labels(data)
    task_ids = set(labels)
    if not task_ids:
        return None, "no per-task labels found — skipped"

    # Group model calls by weave_task_id (matches task ids), ordered by time.
    calls_by_task: dict[str, list] = {}
    for c in calls:
        wt = str(c.get("weave_task_id"))
        calls_by_task.setdefault(wt, []).append(c)
    for lst in calls_by_task.values():
        lst.sort(key=lambda c: c.get("started_at") or "")

    # Determine the model name from the calls (fallback to total_usage).
    model_name = ""
    for c in calls:
        m = (c.get("inputs") or {}).get("model")
        if m:
            model_name = m
            break
    if not model_name:
        model_name = next(iter(data.get("total_usage", {})), agent_name)

    now = datetime.now(timezone.utc).isoformat()
    samples = []
    n_pass = 0
    for tid in sorted(task_ids, key=lambda x: (len(x), x)):
        verdict = labels[tid]
        is_pass = verdict >= 0.5
        n_pass += is_pass
        task_calls = calls_by_task.get(tid, [])

        messages = []
        raw_calls = []
        first_prompt = ""
        for c in task_calls:
            try:
                inp = c.get("inputs") if isinstance(c.get("inputs"), dict) else {}
                in_msgs = inp.get("messages") or []
                # Last message of the request = the actual prompt for this call.
                # Entries are usually dicts, but some agents log lists/strings.
                if in_msgs:
                    last = in_msgs[-1]
                    prompt = _content_to_text(
                        last.get("content") if isinstance(last, dict) else last)
                else:
                    prompt = ""
                answer = _output_text(c.get("output"))
            except Exception:  # noqa: BLE001 - never let one odd call drop a file
                prompt, answer, in_msgs, inp = "", "", [], {}
            exc = c.get("exception")
            if not first_prompt and prompt:
                first_prompt = prompt
            messages.append(ChatMessageUser(content=prompt or "(empty request)"))
            messages.append(ChatMessageAssistant(
                content=answer or (f"[exception] {exc}" if exc else "(no output)")
            ))
            raw_calls.append({
                "model": inp.get("model"),
                "started_at": c.get("started_at"),
                "ended_at": c.get("ended_at"),
                "request_messages": in_msgs,
                "output": c.get("output"),
                "exception": exc,
            })

        samples.append(EvalSample(
            id=tid,
            epoch=1,
            input=(first_prompt[:4000] or f"[{benchmark} task {tid}]"),
            target="PASS",
            messages=messages or [ChatMessageUser(content=f"[{benchmark} task {tid}]")],
            scores={"grader": Score(
                value=verdict,
                answer="PASS" if is_pass else "FAIL",
                explanation=f"HAL {benchmark} label ({label_src})",
                metadata={"latency_s": latencies.get(tid), "n_calls": len(task_calls)},
            )},
            metadata={
                "task_id": tid,
                "benchmark": benchmark,
                "agent_name": agent_name,
                "n_model_calls": len(task_calls),
                "latency_s": latencies.get(tid),
                "raw_calls": raw_calls,
            },
        ))

    total = len(samples)
    acc = res.get("accuracy")
    if acc is None:
        acc = n_pass / total if total else 0.0

    log = EvalLog(
        version=2,
        status="success",
        eval=EvalSpec(
            created=now,
            task=benchmark,
            task_id=f"hal/{benchmark}/{agent_name}",
            task_display_name=f"HAL · {benchmark}",
            dataset=EvalDataset(name=benchmark, location="agent-evals/hal_traces"),
            model=model_name,
            config=EvalConfig(),
            metadata={
                "source": "HAL hal_traces",
                "agent_name": agent_name,
                "run_id": run_id,
                "total_cost": data.get("total_cost"),
                "git_commit": (data.get("git_info") or {}).get("commit"),
            },
        ),
        plan=EvalPlan(name=agent_name),
        results=EvalResults(
            total_samples=total,
            completed_samples=total,
            scores=[EvalScore(
                name="grader", scorer="grader", params={},
                metrics={
                    "accuracy": EvalMetric(name="accuracy", value=float(acc)),
                    "pass_count": EvalMetric(name="pass_count", value=float(n_pass)),
                },
            )],
        ),
        stats=EvalStats(started_at=now, completed_at=now),
        samples=samples,
    )
    return log, f"{n_pass}/{total} pass ({acc:.1%}) | {health} | labels={label_src}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Import HAL agent traces to Inspect .eval")
    ap.add_argument("--zip", action="append", default=[], type=Path, help="Local trace zip (repeatable)")
    ap.add_argument("--dir", type=Path, help="Directory of local trace zips")
    ap.add_argument("--hf-benchmark", default=None,
                    help="Stream from HF: comma-separated benchmark prefixes (empty string = ALL)")
    ap.add_argument("--hf-repo", default=HF_REPO, help="HF dataset repo id")
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/hal_zips"),
                    help="Temp dir for streamed zips (deleted per-file after export)")
    ap.add_argument("--out", type=Path, default=Path("inspect_logs/hal"), help="Output dir")
    ap.add_argument("--min-output-rate", type=float, default=0.5,
                    help="Skip runs where <this fraction of calls have output (default 0.5)")
    ap.add_argument("--allow-slim", action="store_true", help="Also import slim traces (labels only)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = all)")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="Abort before a download if free disk drops below this (default 20 GB)")
    ap.add_argument("--max-zip-gb", type=float, default=0.7,
                    help="Skip traces larger than this (GB); decrypt+parse holds the "
                         "whole JSON in RAM (~10-15x the zip), so huge browser-agent "
                         "traces OOM on modest hosts. 0 = no limit (default 0.7)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Reprocess files even if a .eval/.skipped marker exists")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_err = 0

    def record(stem: str, status: str, note: str, tag: str) -> None:
        nonlocal n_ok, n_skip, n_err
        if status == "ok":
            n_ok += 1
            print(f"  [ok]   {tag}: {note}")
        elif status == "skip":
            n_skip += 1
            (args.out / f"{stem}.skipped").write_text(note)  # marker for resume
            print(f"  [skip] {tag}: {note}")
        else:
            n_err += 1
            print(f"  [ERR]  {tag}: {note}")

    if args.hf_benchmark is not None:
        # Streaming mode: list -> download -> export -> delete, one file at a time.
        prefixes = [p for p in args.hf_benchmark.split(",") if p]
        names = list_hf_zips(args.hf_repo, prefixes)
        if args.limit:
            names = names[: args.limit]
        args.work_dir.mkdir(parents=True, exist_ok=True)
        print(f"HF {args.hf_repo}: {len(names)} zip(s) to consider (prefixes={prefixes or 'ALL'})")
        for i, name in enumerate(names, 1):
            stem = _eval_stem(name)
            if not args.no_resume and (
                (args.out / f"{stem}.eval").exists() or (args.out / f"{stem}.skipped").exists()
            ):
                print(f"  [resume] {i}/{len(names)} {name}: already done")
                continue
            tag = f"{i}/{len(names)} {name}"
            if args.max_zip_gb > 0:
                size_gb = hf_file_size(args.hf_repo, name) / (1024 ** 3)
                if size_gb > args.max_zip_gb:
                    record(stem, "skip",
                           f"too large ({size_gb:.2f} GB > --max-zip-gb "
                           f"{args.max_zip_gb:.2f}); decrypt+parse would OOM — "
                           f"process on a high-RAM host", tag)
                    continue
            avail = free_gb(args.work_dir)
            if avail < args.min_free_gb:
                print(f"\n[ABORT] free disk {avail:.1f} GB < --min-free-gb "
                      f"{args.min_free_gb:.0f} GB. Stopping to protect the disk; "
                      f"rerun to resume once space is freed.")
                break
            dest = args.work_dir / name
            try:
                download_hf_file(args.hf_repo, name, dest)
                status, note = process_zip(dest, args.out, args.min_output_rate,
                                           args.allow_slim, stem)
            except Exception as e:  # noqa: BLE001
                status, note = "err", f"download/process failed: {e}"
            finally:
                dest.unlink(missing_ok=True)  # keep peak disk to one zip
            record(stem, status, note, tag)
    else:
        # Local mode.
        zips = list(args.zip)
        if args.dir:
            zips += sorted(args.dir.glob("*.zip"))
        if not zips:
            ap.error("provide --zip, --dir, or --hf-benchmark")
        for zp in zips:
            stem = _eval_stem(zp.name)
            if not args.no_resume and (args.out / f"{stem}.eval").exists():
                print(f"  [resume] {zp.name}: already done")
                continue
            status, note = process_zip(zp, args.out, args.min_output_rate,
                                       args.allow_slim, stem)
            record(stem, status, note, zp.name)

    print(f"\nDone. {n_ok} imported, {n_skip} skipped, {n_err} errors -> {args.out}")


if __name__ == "__main__":
    main()
