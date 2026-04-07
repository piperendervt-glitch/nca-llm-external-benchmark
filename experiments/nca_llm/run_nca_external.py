"""
External benchmark NCA experiment.
4 conditions: homo_nca / het_nca_v1 / het_nca_v2

Usage:
  python run_nca_external.py --benchmark alphanli --condition homo_nca
  python run_nca_external.py --benchmark ruletaker_d1 --condition het_nca_v1
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import httpx

from benchmark_loader import load_alphanli, load_ruletaker_d1

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CONDITIONS = {
    "homo_nca":   ["qwen2.5:7b", "qwen2.5:7b", "qwen2.5:7b"],
    "het_nca_v1": ["qwen2.5:7b", "llama3:latest", "mistral:7b"],
    "het_nca_v2": ["qwen2.5:7b", "gemma2:7b", "phi3"],
}

AGREE = [30, 80, 80]
STEPS = 3

_client = httpx.Client(timeout=120.0)


# ── LLM call ──────────────────────────────────────────────────────────────────

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


def format_output(output: dict) -> str:
    decision = output.get("decision", "UNKNOWN")
    confidence = output.get("confidence", 0.5)
    reasoning = output.get("reasoning", "")
    return f"{decision} (confidence: {confidence}) - {reasoning}"


def agreement_instruction(agreement_pct: int) -> str:
    if agreement_pct >= 50:
        return (
            f"You should agree with your teammates approximately {agreement_pct}% of the time.\n"
            f"If they strongly agree, consider following their consensus."
        )
    return (
        f"You should agree with your teammates approximately {agreement_pct}% of the time.\n"
        f"If they strongly agree, consider maintaining your independent judgment."
    )


# ── Role prompts ──────────────────────────────────────────────────────────────

def solver_prompt(task_input: str, agreement_pct: int) -> str:
    agree = agreement_instruction(agreement_pct)
    return f"""You are the Solver in a 3-node reasoning network.
Task: {task_input}

Solve this step by step. Show your work clearly.
Determine if the statement is CORRECT or INCORRECT.
{agree}

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Step-by-step solution in 2-3 sentences"
}}"""


def verifier_prompt(task_input: str, solver_output: dict, agreement_pct: int) -> str:
    solver_info = format_output(solver_output)
    agree = agreement_instruction(agreement_pct)
    return f"""You are the Verifier in a 3-node reasoning network.
Task: {task_input}

The Solver's answer: {solver_info}

Independently verify this answer from scratch.
Do NOT simply agree - check the math yourself.
Determine if the statement is CORRECT or INCORRECT.
{agree}

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Your independent verification in 2-3 sentences"
}}"""


def critic_prompt(task_input: str, solver_output: dict, verifier_output: dict,
                  agreement_pct: int) -> str:
    solver_info = format_output(solver_output)
    verifier_info = format_output(verifier_output)
    agree = agreement_instruction(agreement_pct)
    return f"""You are the Critic in a 3-node reasoning network.
Task: {task_input}

Solver's reasoning: {solver_info}
Verifier's reasoning: {verifier_info}

Critically evaluate both answers.
If they disagree, determine who is right.
If they agree but seem wrong, say so.
Determine if the statement is CORRECT or INCORRECT.
{agree}

Respond ONLY in the following JSON format (nothing else):
{{
  "decision": "CORRECT" or "INCORRECT",
  "confidence": 0.0 to 1.0,
  "reasoning": "Final judgment with brief explanation"
}}"""


# ── NCA runner ────────────────────────────────────────────────────────────────

def run_nca(task_input: str, models: list[str]) -> dict:
    all_steps = []

    for step in range(STEPS):
        prompt_s = solver_prompt(task_input, AGREE[0])
        solver_out = call_llm(models[0], prompt_s)

        prompt_v = verifier_prompt(task_input, solver_out, AGREE[1])
        verifier_out = call_llm(models[1], prompt_v)

        prompt_c = critic_prompt(task_input, solver_out, verifier_out, AGREE[2])
        critic_out = call_llm(models[2], prompt_c)

        all_steps.append({
            "step": step,
            "roles": ["solver", "verifier", "critic"],
            "solver": {"node": 0, "model": models[0], "output": solver_out},
            "verifier": {"node": 1, "model": models[1], "output": verifier_out},
            "critic": {"node": 2, "model": models[2], "output": critic_out},
        })

    last = all_steps[-1]
    node_outputs = {
        "solver": {
            "decision": last["solver"]["output"].get("decision", "UNKNOWN"),
            "confidence": float(last["solver"]["output"].get("confidence", 0.5)),
        },
        "verifier": {
            "decision": last["verifier"]["output"].get("decision", "UNKNOWN"),
            "confidence": float(last["verifier"]["output"].get("confidence", 0.5)),
        },
        "critic": {
            "decision": last["critic"]["output"].get("decision", "UNKNOWN"),
            "confidence": float(last["critic"]["output"].get("confidence", 0.5)),
        },
    }

    votes = [node_outputs[r]["decision"] for r in ("solver", "verifier", "critic")]
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
        "steps_data": all_steps,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["alphanli", "ruletaker_d1"])
    parser.add_argument("--condition", required=True, choices=["homo_nca", "het_nca_v1", "het_nca_v2"])
    args = parser.parse_args()

    models = CONDITIONS[args.condition]
    results_dir = REPO_ROOT / "results" / "nca_llm" / args.benchmark
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{args.condition}.jsonl"

    print("=" * 60)
    print(f"NCA External: {args.benchmark} x {args.condition}")
    print(f"Models: {models}")
    print(f"Agree: {AGREE}, Steps: {STEPS}")
    print(f"Output: {results_path}")
    print("=" * 60)

    # Load tasks
    if args.benchmark == "alphanli":
        tasks = load_alphanli()
    else:
        tasks = load_ruletaker_d1()

    # Resume support
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
                result = run_nca(task.question, models)
                verdict = result["verdict"]
                vote_dist = result["vote_distribution"]
                is_unanimous = result["is_unanimous"]
                node_outputs = result["node_outputs"]
                steps_data = result["steps_data"]
            except Exception as e:
                verdict = "ERROR"
                vote_dist = {"CORRECT": 0, "INCORRECT": 0}
                is_unanimous = False
                node_outputs = {}
                steps_data = []
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
                "steps_data": steps_data,
                "condition": args.condition,
                "models": models,
                "elapsed_sec": round(elapsed, 2),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            done += 1
            if is_correct:
                correct_count += 1

            if done % 10 == 0 or done == total:
                acc = correct_count / (done - len(completed)) * 100 if (done - len(completed)) > 0 else 0
                print(f"  [{args.condition}] {done}/{total} ({acc:.1f}% acc) last={elapsed:.1f}s")

    print(f"Complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
