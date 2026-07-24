# refute-inspect

Inspect AI adapter for **[REFUTE](https://huggingface.co/datasets/BGPT-OFFICIAL/refute)** — judge-free tasks for scientific critique and epistemic calibration on recent science paper summaries.

**Why this exists:** AI can sound scientifically careful while quietly turning "may" into "does." REFUTE measures whether models know what the evidence allows, what would overturn a claim, and when confidence is justified.

- Public writeup: https://bgpt.pro/refute
- Leaderboard: https://huggingface.co/spaces/BGPT-OFFICIAL/refute-leaderboard
- Interim preprint package: https://github.com/connerlambden/refute-inspect/releases/tag/v3.0.0-preprint

## Tasks

| Task | Items | Scoring |
|------|-------|---------|
| `refute_knowledge` | 60 | 4-way MCQ (`multiple_choice` + `choice`; chance = 25%) |
| `refute_soundness` | 74 | Binary sound/flawed (`generate` + custom scorer) |
| `refute_forced_choice` | 37 pairs | Paired A/B flaw discrimination (chance = 50%) |

Generative critique tasks (`refute_120`, `refute_hard_60`) require an LLM judge and are omitted here.

## Install

```bash
pip install inspect-ai datasets
git clone https://github.com/connerlambden/refute-inspect.git
cd refute-inspect && pip install -e .
```

## Run

```bash
inspect eval refute_inspect/refute_knowledge --model openai/gpt-4o
inspect eval refute_inspect/refute_soundness --model openai/gpt-4o
inspect eval refute_inspect/refute_forced_choice --model openai/gpt-4o
```

Smoke test without API keys:

```bash
inspect eval refute_inspect/refute_knowledge --model mockllm/model --limit 1
```

Dataset loads from Hugging Face `BGPT-OFFICIAL/refute` at revision `2be5046d4097fc213ea8ba7e193719b8da096169`. Override with `REFUTE_KNOWLEDGE_JSONL` or `REFUTE_SOUNDNESS_JSONL` for local JSONL fixtures.

## Register with Inspect Evals

Direct PRs to [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) are no longer accepted. Register this package via the [Inspect Evals Register](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/register/README.md) — one **Register Eval Submission** issue per task.

## Links

- Dataset: https://huggingface.co/datasets/BGPT-OFFICIAL/refute
- Technical report: https://huggingface.co/datasets/BGPT-OFFICIAL/refute/blob/main/TECHNICAL_REPORT.md
- Leaderboard: https://huggingface.co/spaces/BGPT-OFFICIAL/refute-leaderboard
- Hub integrator index: https://huggingface.co/datasets/BGPT-OFFICIAL/refute/blob/main/INTEGRATORS.md
