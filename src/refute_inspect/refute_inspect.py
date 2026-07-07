"""Inspect AI adapter for the REFUTE benchmark (judge-free core tasks).

REFUTE (Reasoning Over Evidence — Falsification, Uncertainty, Truth-grounding &
Epistemics) evaluates whether language models critique recent science papers with
calibrated, evidence-grounded judgment.

This adapter exposes three *objective, judge-free* REFUTE tasks so any model can be
scored with a single command and no LLM judge:

  * refute_knowledge      — closed-book 4-way MCQ on recent findings (chance = 25%)
  * refute_forced_choice  — pick the more flawed of twin summaries (chance = 50%)
  * refute_soundness      — binary sound/flawed classification over single summaries

The generative critique tasks (refute_hard_60 / refute_120) are rubric-scored and need
an LLM judge; they are intentionally not included here so this adapter stays fully
objective and reproducible.

Usage
-----
    pip install inspect-ai datasets
    inspect eval refute_inspect/refute_knowledge --model openai/gpt-4o
    inspect eval refute_inspect/refute_forced_choice --model openai/gpt-4o
    inspect eval refute_inspect/refute_soundness --model openai/gpt-4o

By default datasets load from Hugging Face ``BGPT-OFFICIAL/refute`` at a pinned
revision. Override with ``REFUTE_<CONFIG>_JSONL`` env vars for local JSONL files.

Dataset: https://huggingface.co/datasets/BGPT-OFFICIAL/refute
Leaderboard: https://huggingface.co/spaces/BGPT-OFFICIAL/refute-leaderboard
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, choice, scorer, stderr
from inspect_ai.solver import TaskState, generate, multiple_choice, system_message

HF_DATASET = "BGPT-OFFICIAL/refute"
HF_REVISION = "2be5046d4097fc213ea8ba7e193719b8da096169"

FORCED_CHOICE_SYSTEM = (
    "You are a rigorous methodological reviewer. You will see two study summaries, A and B, "
    "on the same topic. Exactly ONE has a more serious methodological flaw that undermines its "
    "central conclusion. Decide which. Respond on one line in EXACTLY this format: "
    "'ANSWER=<A or B>' then at most one short clause naming the flaw."
)

SOUNDNESS_SYSTEM = (
    "You are a rigorous methodological reviewer. You will see one study summary. Decide whether its "
    "central conclusion is methodologically SOUND or FLAWED given what is described. Respond on one "
    "line in EXACTLY this format: 'LABEL=<sound or flawed>' then at most one short clause naming the "
    "flaw if any."
)

KNOWLEDGE_SYSTEM = (
    "You are answering a closed-book scientific knowledge question about a recent empirical study. "
    "Pick the option that best matches the reported finding."
)

_ANSWER_RE = re.compile(r"ANSWER\s*=\s*([AB])", re.I)
_FALLBACK_AB_RE = re.compile(r"\b([AB])\b")
_LABEL_RE = re.compile(r"LABEL\s*=\s*(sound|flawed)", re.I)
_FALLBACK_LABEL_RE = re.compile(r"\b(sound|flawed|unsound)\b", re.I)


def _base_of(item_id: str) -> str:
    """Strip the -clean / -flaw suffix to recover the shared twin id."""
    return item_id.rsplit("-clean", 1)[0].rsplit("-flaw", 1)[0]


def _parse_options(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


_ENV_JSONL = {
    "refute_knowledge": "REFUTE_KNOWLEDGE_JSONL",
    "refute_soundness": "REFUTE_SOUNDNESS_JSONL",
}


def _load_config(config: str) -> list[dict[str, Any]]:
    """Load a REFUTE config from a local JSONL override or the HF Hub."""
    local = os.environ.get(_ENV_JSONL.get(config, f"REFUTE_{config.upper()}_JSONL"))
    if local:
        with open(local, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Install `datasets` (pip install datasets) or set "
            f"REFUTE_{config.upper()}_JSONL to a local JSONL file."
        ) from exc
    ds = load_dataset(HF_DATASET, config, split="train", revision=HF_REVISION)
    return [dict(row) for row in ds]


def record_to_knowledge_sample(record: dict[str, Any]) -> Sample:
    options = _parse_options(record.get("options"))
    letters = sorted(options.keys())
    return Sample(
        input=record["prompt"],
        choices=letters,
        target=str(record["answer"]).upper(),
        id=record["id"],
        metadata={"task": record.get("task", "knowledge")},
    )


def _build_forced_choice_samples(items: list[dict[str, Any]]) -> list[Sample]:
    by_base: dict[str, dict[str, dict]] = {}
    for it in items:
        by_base.setdefault(_base_of(it["id"]), {})[it["label"]] = it

    samples: list[Sample] = []
    for base in sorted(by_base):
        twin = by_base[base]
        if "sound" not in twin or "flawed" not in twin:
            continue
        flaw_first = int(hashlib.sha1(base.encode()).hexdigest(), 16) % 2 == 0
        a, b = (twin["flawed"], twin["sound"]) if flaw_first else (twin["sound"], twin["flawed"])
        flaw_letter = "A" if flaw_first else "B"
        prompt = (
            f"STUDY A:\n{a['summary']}\n\n"
            f"STUDY B:\n{b['summary']}\n\n"
            "Which study (A or B) has the more serious methodological flaw?"
        )
        samples.append(
            Sample(
                input=prompt,
                target=flaw_letter,
                id=base,
                metadata={"flaw_type": twin["flawed"].get("flaw_type", "")},
            )
        )
    return samples


def _build_soundness_samples(items: list[dict[str, Any]]) -> list[Sample]:
    samples: list[Sample] = []
    for it in items:
        label = str(it["label"]).lower()
        if label not in {"sound", "flawed"}:
            continue
        samples.append(
            Sample(
                input=f"STUDY SUMMARY:\n{it['summary']}\n\nIs this study's central conclusion sound or flawed?",
                target=label,
                id=it["id"],
                metadata={"flaw_type": it.get("flaw_type", "")},
            )
        )
    return samples


@scorer(metrics=[accuracy(), stderr()])
def _forced_choice_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        text = (state.output.completion or "").strip()
        m = _ANSWER_RE.search(text)
        if not m:
            m = _FALLBACK_AB_RE.search(text[:8])
        choice_letter = m.group(1).upper() if m else None
        value = CORRECT if choice_letter is not None and choice_letter == target.text.upper() else INCORRECT
        return Score(value=value, answer=choice_letter, explanation=text)

    return score


@scorer(metrics=[accuracy(), stderr()])
def _soundness_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        text = (state.output.completion or "").strip()
        m = _LABEL_RE.search(text)
        label = m.group(1).lower() if m else None
        if label is None:
            fm = _FALLBACK_LABEL_RE.search(text)
            if fm:
                token = fm.group(1).lower()
                label = "flawed" if token in {"flawed", "unsound"} else "sound"
        value = CORRECT if label is not None and label == target.text.lower() else INCORRECT
        return Score(value=value, answer=label, explanation=text)

    return score


@task
def refute_knowledge() -> Task:
    """REFUTE closed-book knowledge MCQ (60 items; chance = 25%)."""
    rows = _load_config("refute_knowledge")
    samples = [record_to_knowledge_sample(r) for r in rows]
    return Task(
        dataset=MemoryDataset(samples, name="refute_knowledge"),
        solver=[system_message(KNOWLEDGE_SYSTEM), multiple_choice()],
        scorer=choice(),
    )


@task
def refute_forced_choice() -> Task:
    """REFUTE forced-choice soundness task (contamination-proof; chance = 50%)."""
    samples = _build_forced_choice_samples(_load_config("refute_soundness"))
    return Task(
        dataset=MemoryDataset(samples, name="refute_forced_choice"),
        solver=[system_message(FORCED_CHOICE_SYSTEM), generate()],
        scorer=_forced_choice_scorer(),
        config={"temperature": 0.0, "max_tokens": 120},
    )


@task
def refute_soundness() -> Task:
    """REFUTE binary sound/flawed classification task."""
    samples = _build_soundness_samples(_load_config("refute_soundness"))
    return Task(
        dataset=MemoryDataset(samples, name="refute_soundness"),
        solver=[system_message(SOUNDNESS_SYSTEM), generate()],
        scorer=_soundness_scorer(),
        config={"temperature": 0.0, "max_tokens": 120},
    )
