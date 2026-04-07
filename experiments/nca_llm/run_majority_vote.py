"""
Majority vote baseline (no NCA communication).

Usage:
  python run_majority_vote.py --benchmark alphanli
  python run_majority_vote.py --benchmark ruletaker_d1

Models: qwen2.5:7b / llama3:latest / mistral:7b (same as het_nca_v1)
Each model independently solves the task, then majority vote decides.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import httpx

from benchmark_loader import load_alphanli, load_ruletaker_d1

OLLAMA_URL = "http://localhost:11434/api/generate"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ["qwen2.5:7b", "llama3:latest", "mistral:7b"]

_client = httpx.Client(timeout=120.0)


def call_llm(model: str, prompt: str) -> dict:
    response = _client.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"decision": "UNKNOWN", "confidence": 0.5, "reasoning": raw}


def independent_prompt(task_input: str) -> str:
    return f"""You are an independent reasoning agent.
Task: {task_input}

Solve this step by step. Show your work clearly.
Determine if the statement is CORRECT or INCORRECT.

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Step-by-step solution in 2-3 sentences"
}}"""


def run_majority_vote(task_input: str) -> dict:
    """Each model solves independently, then majority vote."""
    outputs = {}
    for model in MODELS:
        prompt = independent_prompt(task_input)
        outputs[model] = call_llm(model, prompt)

    votes = []
    node_outputs = {}
    for i, model in enumerate(MODELS):
        decision = outputs[model].get("decision", "UNKNOWN")
        confidence = float(outputs[model].get("confidence", 0.5))
        node_outputs[f"node_{i}"] = {
            "model": model,
            "decision": decision,
            "confidence": confidence,
        }
        votes.append(decision)

    vote_dist = {
        "CORRECT": votes.count("CORRECT"),
        "INCORRECT": votes.count("INCORRECT"),
    }

    filtered = [v for v in votes if v in ("CORRECT", "INCORRECT")]
    if not filtered:
        verdict = "UNKNOWN"
    else:
        verdict = Counter(filtered).most_common(1)[0][0]

    is_unanimous = any(v == 3 for v in vote_dist.values())

    return {
        "verdict": verdict,
        "vote_distribution": vote_dist,
        "is_unanimous": is_unanimous,
        "node_outputs": node_outputs,
    }


def verdict_matches(verdict: str, label: bool) -> bool:
    if verdict == "CORRECT":
        return label is True
    elif verdict == "INCORRECT":
        return label is False
    return False


def load_completed(path: Path) -> set:
    completed = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    completed.add(r["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["alphanli", "ruletaker_d1"])
    args = parser.parse_args()

    results_dir = REPO_ROOT / "results" / "nca_llm" / args.benchmark
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "majority_vote.jsonl"

    print("=" * 60)
    print(f"Majority Vote: {args.benchmark}")
    print(f"Models: {MODELS}")
    print(f"Output: {results_path}")
    print("=" * 60)

    if args.benchmark == "alphanli":
        tasks = load_alphanli()
    else:
        tasks = load_ruletaker_d1()

    completed = load_completed(results_path)
    remaining = [t for t in tasks if t.task_id not in completed]

    if not remaining:
        print(f"All {len(tasks)} tasks already completed. Skipping.")
        return

    done = len(completed)
    total = len(tasks)
    correct_count = 0
    print(f"{done}/{total} done, {len(remaining)} remaining.")

    mode = "a" if completed else "w"
    with open(results_path, mode, encoding="utf-8") as f_out:
        for i, task in enumerate(remaining):
            t0 = time.time()
            try:
                result = run_majority_vote(task.question)
                verdict = result["verdict"]
                vote_dist = result["vote_distribution"]
                is_unanimous = result["is_unanimous"]
                node_outputs = result["node_outputs"]
            except Exception as e:
                verdict = "ERROR"
                vote_dist = {"CORRECT": 0, "INCORRECT": 0}
                is_unanimous = False
                node_outputs = {}
                print(f"  ERROR on {task.task_id}: {e}")

            elapsed = time.time() - t0
            is_correct = verdict_matches(verdict, task.label)

            record = {
                "task_id": task.task_id,
                "benchmark": task.benchmark,
                "question": task.question,
                "label": task.label,
                "prediction": verdict,
                "is_correct": is_correct,
                "vote_distribution": vote_dist,
                "is_unanimous": is_unanimous,
                "node_outputs": node_outputs,
                "condition": "majority_vote",
                "models": MODELS,
                "elapsed_sec": round(elapsed, 2),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            done += 1
            if is_correct:
                correct_count += 1

            if done % 10 == 0 or done == total:
                acc = correct_count / (done - len(completed)) * 100 if (done - len(completed)) > 0 else 0
                print(f"  [majority_vote] {done}/{total} ({acc:.1f}% acc) last={elapsed:.1f}s")

    print(f"Complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
