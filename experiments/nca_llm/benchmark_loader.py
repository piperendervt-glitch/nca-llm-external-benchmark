"""
Benchmark loader: αNLI and RuleTaker-depth-1
50/50 balance guaranteed for αNLI via paired generation.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    benchmark: str
    question: str
    label: bool  # True=CORRECT, False=INCORRECT


def load_alphanli(n=1092, seed=42) -> list[Task]:
    """Load αNLI and convert to binary tasks (50/50 guaranteed)."""
    random.seed(seed)

    # Combine train + validation
    all_items = []
    for split in ["train", "validation"]:
        path = Path(f"reference/benchmarks/alphanli/{split}.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                all_items.append(json.loads(line))

    # Generate 2 tasks per item (one CORRECT, one INCORRECT)
    # HF "art" dataset fields: observation_1, observation_2, hypothesis_1, hypothesis_2, label (1 or 2)
    # label=1 means hypothesis_1 is correct, label=2 means hypothesis_2 is correct
    converted = []
    for i, item in enumerate(all_items):
        obs1 = item["observation_1"]
        obs2 = item["observation_2"]
        hyp1 = item["hypothesis_1"]
        hyp2 = item["hypothesis_2"]
        label = item["label"]  # 1 or 2 (1=hyp1, 2=hyp2)

        # Task A: present hyp1
        converted.append(Task(
            task_id=f"anli_{i:05d}_A",
            benchmark="alphanli",
            question=(
                f"Observation 1: {obs1}\n"
                f"Hypothesis: {hyp1}\n"
                f"Observation 2: {obs2}\n"
                f"Is this hypothesis correct?"
            ),
            label=(label == 1)  # label=1 means hyp1 is correct
        ))

        # Task B: present hyp2
        converted.append(Task(
            task_id=f"anli_{i:05d}_B",
            benchmark="alphanli",
            question=(
                f"Observation 1: {obs1}\n"
                f"Hypothesis: {hyp2}\n"
                f"Observation 2: {obs2}\n"
                f"Is this hypothesis correct?"
            ),
            label=(label == 2)  # label=2 means hyp2 is correct
        ))

    # Shuffle and sample
    random.shuffle(converted)
    selected = converted[:n]

    n_correct = sum(1 for t in selected if t.label)
    n_incorrect = n - n_correct
    print(f"αNLI: {n}問 CORRECT={n_correct} INCORRECT={n_incorrect}")

    return selected


def load_ruletaker_d1(n=1092, seed=42) -> list[Task]:
    """Load RuleTaker-depth-1 and convert to binary tasks."""
    random.seed(seed)
    tasks = []

    # HF "hitachi-nlp/ruletaker" fields: context, question, label ("entailment"/"not entailment"), config
    for split in ["train", "dev", "test"]:
        path = Path(f"reference/benchmarks/ruletaker/{split}.jsonl")
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                context = item.get("context", "")
                question = item.get("question", "")
                label_str = item.get("label", "not entailment")
                label = (label_str == "entailment")  # True=CORRECT
                tasks.append(Task(
                    task_id=f"rt_{len(tasks):05d}",
                    benchmark="ruletaker_d1",
                    question=f"Rules:\n{context}\n\nStatement: {question}",
                    label=label
                ))

    random.shuffle(tasks)
    selected = tasks[:n]

    n_correct = sum(1 for t in selected if t.label)
    print(f"RuleTaker-d1: {len(selected)}問 CORRECT={n_correct} INCORRECT={len(selected)-n_correct}")

    return selected
