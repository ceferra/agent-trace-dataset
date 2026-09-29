# Agent-Execution Trace Dataset

A dataset of AI-agent execution traces, logged in [Inspect AI](https://inspect.aisi.org.uk/) `.eval`
format, collected to study the reliability and predictability of autonomous LLM agents: given a task
and a model, can we anticipate — before or during execution — whether the agent will succeed or fail?

Traces were produced by running the [OpenClaw](https://github.com/) agent inside the
**Real-Claw-Bench (RCB)** harness against a range of LLMs, both via commercial APIs and served locally
as open weights. See [`docs/Agent_Trace_Dataset_Provenance.pdf`](docs/Agent_Trace_Dataset_Provenance.pdf)
for the full provenance report (sources, collection method, benchmark coverage, known gaps).

## What's in this repo

| Path | Contents |
|---|---|
| `data/api-models/tbench/` | RCB harness vs. 8 commercial-API models (Azure OpenAI: GPT-4o, GPT-4.1, GPT-5.4, o1; Azure AI Foundry: Llama-4-Maverick, Mistral-Large-3, DeepSeek-v3.2, Grok-4.3), on Terminal-Bench-Core, SWE-bench-Verified, QuixBugs and Aider-Polyglot |
| `data/api-models/azure50/`, `azure_v2/`, `azure_v3/` | The same 8 models on RCB's own 281-task native suite |
| `data/open-weight-models/` | The same harness vs. locally served open-weight models (Qwen2.5 7B/14B/32B, Qwen2.5-Coder 7B/14B/32B, Mistral-Nemo-12B, Mistral-Small-22B, DeepSeek-Coder-V2-Lite), run on the DSIC SLURM cluster with vLLM |
| `scripts/azure_openai_proxy.py` | Local proxy translating OpenClaw's OpenAI-style requests into the Azure OpenAI / Azure AI Foundry API convention (reasoning-model support, retry/backoff) |
| `scripts/export_inspect.py`, `export_inspect_tb.py` | Export RCB run logs into the Inspect AI `.eval` format |
| `scripts/import_hal_traces.py` | Pipeline to download, decrypt and convert traces from Princeton's [Holistic Agent Leaderboard](https://hal.cs.princeton.edu/) (HAL) into the same `.eval` format |
| `docs/Agent_Trace_Dataset_Provenance.pdf` | Full provenance report |

## A note on the HAL-derived traces (not included here)

A third source, external to this repo, exists: traces imported from HAL's own published dataset
(`agent-evals/hal_traces` on Hugging Face — 334 additional `.eval` files across 9 further benchmarks,
16 model families, ~4.9 GB). Those traces are **not redistributed in this repository**, since no
explicit license is published alongside the source dataset at the time of writing. The import script
(`scripts/import_hal_traces.py`) is included so anyone with access to the original Hugging Face dataset
can reproduce that part of the pipeline themselves, under whatever terms apply to that upstream source.
See the provenance report for full details on that source (Source C).

## Format

Every `.eval` file is a self-contained Inspect AI evaluation log: for each task, it preserves the
complete raw sequence of LLM requests and responses (system/user/assistant/tool messages exactly as
sent and received), tool calls and their outputs, per-call timestamps, and the final pass/fail outcome
assigned by the benchmark's own grader. Browse any file with:

```bash
pip install inspect-ai
inspect view data/api-models/tbench/swebench/gpt-4o_inference.eval
```

## Known gaps

- Aider-Polyglot: Mistral-Large-3 (128/225) and Grok-4.3 (3/225) are incomplete (an Azure Foundry key
  expired mid-run and was not renewed).
- USACO and AppWorld-dev were attempted under this harness but dropped as impractically slow, and are
  not included here.
- Open-weight-model traces are smaller partial runs (not the full 281-task RCB suite) used mainly to
  validate the harness before it was migrated to API-served models.

## License

Code in `scripts/` is released under the MIT license (see [`LICENSE`](LICENSE)). The trace data in
`data/` was produced entirely in-house and carries no third-party restrictions.
