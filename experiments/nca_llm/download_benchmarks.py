"""Download αNLI and RuleTaker-depth-1 from HuggingFace."""

from datasets import load_dataset
import json
from pathlib import Path

# αNLI (HF dataset name: "art")
print("Downloading αNLI...")
dataset = load_dataset("art")
path = Path("reference/benchmarks/alphanli")
path.mkdir(parents=True, exist_ok=True)
for split in ["train", "validation"]:
    with open(path / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for item in dataset[split]:
            # Fields: observation_1, observation_2, hypothesis_1, hypothesis_2, label
            f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
print(f"αNLI: {len(dataset['train'])} train + {len(dataset['validation'])} val")

# RuleTaker depth-1 (HF: "hitachi-nlp/ruletaker", filter config=="depth-1")
print("Downloading RuleTaker...")
dataset = load_dataset("hitachi-nlp/ruletaker")
path = Path("reference/benchmarks/ruletaker")
path.mkdir(parents=True, exist_ok=True)
split_map = {"train": "train", "dev": "dev", "test": "test"}
total = 0
for split_name, hf_split in split_map.items():
    items = [x for x in dataset[hf_split] if x["config"] == "depth-1"]
    with open(path / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            # Fields: context, question, label ("entailment"/"not entailment"), config
            f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
    total += len(items)
    print(f"  RuleTaker-d1 {split_name}: {len(items)} items")
print(f"RuleTaker-d1 total: {total}")
