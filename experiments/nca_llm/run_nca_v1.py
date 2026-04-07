"""
run_nca_v1.py

MVE-20260404-01: NCA-LLM experiment (3 nodes, steps=3).
Solver/Verifier/Critic with fixed roles.
Aggregation: simple majority (Counter).
"""

import json
import time
from collections import Counter
from pathlib import Path

import httpx

from task_generator import generate_tasks as generate_world_consistency_tasks

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "nca_llm" / "v1"
RESULTS_PATH = RESULTS_DIR / "nca_v1_results.jsonl"

MODELS = ["qwen2.5:7b", "llama3:latest", "mistral:7b"]
AGREE = [30, 80, 80]
STEPS = 3

_client = httpx.Client(timeout=120.0)


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(model: str, prompt: str) -> dict:
    response = _client.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
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

def run_nca(task_input: str) -> dict:
    """Run 3-node NCA with fixed roles for the configured number of steps."""
    all_steps = []

    for step in range(STEPS):
        # Solver (node 0: qwen2.5:7b)
        prompt_s = solver_prompt(task_input, AGREE[0])
        solver_out = call_llm(MODELS[0], prompt_s)

        # Verifier (node 1: llama3) sees solver output
        prompt_v = verifier_prompt(task_input, solver_out, AGREE[1])
        verifier_out = call_llm(MODELS[1], prompt_v)

        # Critic (node 2: mistral) sees both
        prompt_c = critic_prompt(task_input, solver_out, verifier_out, AGREE[2])
        critic_out = call_llm(MODELS[2], prompt_c)

        all_steps.append({
            "step": step,
            "roles": ["solver", "verifier", "critic"],
            "solver": {"node": 0, "model": MODELS[0], "output": solver_out},
            "verifier": {"node": 1, "model": MODELS[1], "output": verifier_out},
            "critic": {"node": 2, "model": MODELS[2], "output": critic_out},
        })

    # Extract votes from last step
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

    # Simple majority aggregation (Counter)
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
    if verdict in ("CORRECT", "CONSISTENT"):
        return label is True
    elif verdict in ("INCORRECT", "CONTRADICTION"):
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NCA-LLM v1: 3 nodes, steps=3, fixed roles")
    print(f"Models: {MODELS}")
    print(f"Agree: {AGREE}")
    print(f"Aggregation: simple majority (Counter)")
    print(f"Output: {RESULTS_PATH}")
    print("=" * 60)

    # Load tasks
    raw_tasks = generate_world_consistency_tasks()
    tasks = []
    for t in raw_tasks:
        tasks.append({
            "task_id": f"wc_{t.task_id:03d}",
            "task_set": "world_consistency",
            "task_type": "world_consistency",
            "question": t.question,
            "world_rule": t.world_rule,
            "label": t.label,
            "task_input": f"World rule: {t.world_rule}\nStatement: {t.question}",
        })

    print(f"Loaded {len(tasks)} tasks")

    # Resume support
    completed = load_completed(RESULTS_PATH)
    remaining = [t for t in tasks if t["task_id"] not in completed]

    if not remaining:
        print(f"All {len(tasks)} tasks already completed. Skipping.")
        return

    done = len(completed)
    total = len(tasks)
    correct_count = 0
    print(f"{done}/{total} done, {len(remaining)} remaining.")

    mode = "a" if completed else "w"
    with open(RESULTS_PATH, mode, encoding="utf-8") as f_out:
        for i, task in enumerate(remaining):
            t0 = time.time()
            try:
                result = run_nca(task["task_input"])
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
                print(f"  ERROR on {task['task_id']}: {e}")

            elapsed = time.time() - t0
            is_correct = verdict_matches(verdict, task["label"])

            record = {
                "task_id": task["task_id"],
                "task_set": task["task_set"],
                "task_type": task["task_type"],
                "question": task["question"],
                "label": task["label"],
                "prediction": verdict,
                "is_correct": is_correct,
                "vote_distribution": vote_dist,
                "is_unanimous": is_unanimous,
                "node_outputs": node_outputs,
                "steps_data": steps_data,
                "elapsed_sec": round(elapsed, 2),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            done += 1
            if is_correct:
                correct_count += 1

            if done % 10 == 0 or done == total:
                acc = correct_count / (done - len(completed)) * 100 if (done - len(completed)) > 0 else 0
                print(f"  [NCA] {done}/{total} ({acc:.1f}% acc) last={elapsed:.1f}s")

    print(f"Complete. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
