"""
run_nca_stigmergy.py

Stigmergy-NCA experiment.
Extends run_nca_external.py with a pheromone layer.

Each node reads the pheromone state before reasoning,
then writes back its confidence as a pheromone trace.
The final aggregation is pheromone-weighted majority vote.

New condition added: stigmergy_nca (uses het_nca_v1 models)

Usage:
  python run_nca_stigmergy.py --benchmark alphanli --condition stigmergy_nca
  python run_nca_stigmergy.py --benchmark ruletaker_d1 --condition stigmergy_nca

Compare against existing conditions:
  python run_nca_external.py --benchmark alphanli --condition het_nca_v1
  python run_nca_external.py --benchmark alphanli --condition homo_nca
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import httpx

from benchmark_loader import load_alphanli, load_ruletaker_d1

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CONDITIONS = {
    # Existing conditions (replicated for self-contained comparison)
    "homo_nca":        ["qwen2.5:7b", "qwen2.5:7b", "qwen2.5:7b"],
    "het_nca_v1":      ["qwen2.5:7b", "llama3:latest", "mistral:7b"],
    "het_nca_v2":      ["qwen2.5:7b", "gemma2:9b", "phi3"],
    # New: stigmergy condition (same models as het_nca_v1 for direct comparison)
    "stigmergy_nca":   ["qwen2.5:7b", "llama3:latest", "mistral:7b"],
}

STEPS = 3
PHEROMONE_DECAY = 0.3   # How much previous pheromone persists each step (0~1)
PHEROMONE_INIT  = 0.5   # Neutral starting value

_client = httpx.Client(timeout=120.0)


# ── LLM call ─────────────────────────────────────────────────────────────────

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


# ── Pheromone layer ───────────────────────────────────────────────────────────

class PheromoneLayer:
    """
    Tracks confidence traces left by each node.

    pheromone[i] represents the accumulated evidence
    that node i's decision is in the "A" direction.

    Values above 0.5 → strong pull toward "1" (first option)
    Values below 0.5 → strong pull toward "2" (second option)
    """

    def __init__(self, n_nodes: int, decay: float = PHEROMONE_DECAY):
        self.n = n_nodes
        self.decay = decay
        self.values = [PHEROMONE_INIT] * n_nodes  # per-node pheromone
        self.global_pheromone = PHEROMONE_INIT    # aggregated signal

    def write(self, node_idx: int, decision: str, confidence: float):
        """Node writes its trace after making a decision."""
        # Map decision to direction: "1" → push toward 1.0, "2" → push toward 0.0
        if decision == "1":
            trace = confidence
        elif decision == "2":
            trace = 1.0 - confidence
        else:
            trace = PHEROMONE_INIT  # UNKNOWN → neutral

        # Decay old value, add new trace
        self.values[node_idx] = (
            self.decay * self.values[node_idx] + (1 - self.decay) * trace
        )

        # Update global pheromone as mean of all node pheromones
        self.global_pheromone = sum(self.values) / self.n

    def read_summary(self) -> str:
        """
        Returns a natural language summary of the current pheromone state
        for injection into the next node's prompt.
        Includes direction and strength explicitly.
        """
        g = self.global_pheromone
        strength = abs(g - 0.5) * 2  # 0.0(neutral) ~ 1.0(decisive)

        if g > 0.65:
            direction = "answer 1"
            signal = f"Previous reasoners leaned toward answer 1 (strength: {strength:.2f})."
        elif g < 0.35:
            direction = "answer 2"
            signal = f"Previous reasoners leaned toward answer 2 (strength: {strength:.2f})."
        elif g > 0.55:
            signal = f"Previous reasoners slightly leaned toward answer 1 (strength: {strength:.2f})."
        elif g < 0.45:
            signal = f"Previous reasoners slightly leaned toward answer 2 (strength: {strength:.2f})."
        else:
            signal = f"Previous reasoners showed no clear consensus (strength: {strength:.2f})."

        per_node = ", ".join(
            f"node{i+1}={v:.2f}" for i, v in enumerate(self.values)
        )
        return f"[Pheromone signal: {signal} ({per_node})]"

    def weighted_vote(self, decisions: list[str], confidences: list[float]) -> str:
        """
        Pheromone-weighted aggregation.
        Node votes are weighted by their pheromone consistency:
        nodes whose pheromone aligns with global signal get more weight.
        """
        weights = []
        for i, (dec, conf) in enumerate(zip(decisions, confidences)):
            p = self.values[i]
            # Consistency: how aligned is this node's pheromone with global?
            alignment = 1.0 - abs(p - self.global_pheromone)
            weight = conf * (0.5 + 0.5 * alignment)  # base=conf, bonus from alignment
            weights.append(weight)

        # Weighted vote
        score_1 = sum(w for d, w in zip(decisions, weights) if d == "1")
        score_2 = sum(w for d, w in zip(decisions, weights) if d == "2")

        if score_1 == score_2:
            # Fallback: use global pheromone direction
            return "1" if self.global_pheromone >= 0.5 else "2"
        return "1" if score_1 > score_2 else "2"


# ── Prompt builders ───────────────────────────────────────────────────────────

def label_to_decision(label) -> str:
    """Convert Task.label to decision string "1" or "2".

    alphanli : label=True  → "1" (correct), label=False → "2" (incorrect)
    ruletaker: label=True  → "1" (TRUE),    label=False → "2" (FALSE)
    Both benchmarks use the same True/False convention.
    """
    if label is True or label == "True" or label == 1:
        return "1"
    elif label is False or label == "False" or label == 0:
        return "2"
    return str(label)



def build_base_prompt(task) -> str:
    """Shared prompt structure for all benchmarks.
    task is a Task dataclass with fields: benchmark, label, question, task_id
    """
    benchmark = task.benchmark

    if benchmark == "alphanli":
        return (
            f"{task.question}\n\n"
            "Answer 1: Yes, the hypothesis is correct.\n"
            "Answer 2: No, the hypothesis is not correct.\n"
            'Respond in JSON: {"decision": "1" or "2", "confidence": 0.0-1.0, "reasoning": "..."}'
        )
    elif benchmark == "ruletaker_d1":
        return (
            f"{task.question}\n\n"
            "Answer 1: TRUE\n"
            "Answer 2: FALSE\n"
            'Respond in JSON: {"decision": "1" (TRUE) or "2" (FALSE), "confidence": 0.0-1.0, "reasoning": "..."}'
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def build_stigmergy_prompt(
    task: dict,
    pheromone: PheromoneLayer,
    step: int,
    previous_outputs: list[dict],
) -> str:
    """Prompt with pheromone signal and previous outputs injected."""
    base = build_base_prompt(task)
    pheromone_signal = pheromone.read_summary()

    # Inject pheromone signal
    prompt = f"{pheromone_signal}\n\n{base}"

    # Inject previous reasoning (same as existing NCA approach)
    if previous_outputs:
        prev_text = "\n".join(
            f"  Node {i+1}: decision={o.get('decision','?')}, "
            f"confidence={o.get('confidence', 0.5):.2f}"
            for i, o in enumerate(previous_outputs)
        )
        prompt += f"\n\nPrevious nodes:\n{prev_text}"

    return prompt


# ── NCA run (standard, no pheromone) ─────────────────────────────────────────

def run_nca_standard(task: dict, models: list[str]) -> dict:
    """Baseline NCA: majority vote without pheromone layer."""
    outputs = []
    for step, model in enumerate(models):
        base = build_base_prompt(task)
        if outputs:
            prev_text = "\n".join(
                f"  Node {i+1}: decision={o.get('decision','?')}"
                for i, o in enumerate(outputs)
            )
            prompt = base + f"\n\nPrevious nodes:\n{prev_text}"
        else:
            prompt = base

        out = call_llm(model, prompt)
        outputs.append(out)

    decisions = [o.get("decision", "UNKNOWN") for o in outputs]
    counts = Counter(d for d in decisions if d != "UNKNOWN")
    final = counts.most_common(1)[0][0] if counts else "UNKNOWN"

    return {
        "final_decision": final,
        "decisions": decisions,
        "outputs": outputs,
        "method": "majority_vote",
    }


# ── NCA run (stigmergy) ───────────────────────────────────────────────────────

def run_nca_stigmergy(task, models: list[str], pheromone_mode: str = "dynamic") -> dict:
    """Stigmergy NCA: pheromone-mediated coordination."""
    pheromone = PheromoneLayer(n_nodes=len(models))
    outputs = []
    pheromone_history = []

    for step, model in enumerate(models):
        if pheromone_mode == "no_prompt":
            # 条件C: フェロモンプロンプトなし、weighted_voteのみ
            prompt = build_base_prompt(task)
            if outputs:
                prev_text = "\n".join(
                    f"  Node {i+1}: decision={o.get('decision','?')}"
                    for i, o in enumerate(outputs)
                )
                prompt += f"\n\nPrevious nodes:\n{prev_text}"
        else:
            # dynamic / random / prompt_only: フェロモンプロンプトあり
            prompt = build_stigmergy_prompt(
                task=task,
                pheromone=pheromone,
                step=step,
                previous_outputs=outputs,
            )

        out = call_llm(model, prompt)
        outputs.append(out)

        # Write pheromone trace
        decision = out.get("decision", "UNKNOWN")
        confidence = float(out.get("confidence", 0.5))

        if pheromone_mode in ("random", "prompt_only"):
            # ランダムフェロモン: 実際の推論と無関係な値を注入
            rand_decision = random.choice(["1", "2"])
            rand_confidence = random.uniform(0.5, 0.95)
            pheromone.write(step, rand_decision, rand_confidence)
        else:
            # dynamic / no_prompt: 通常のフェロモン更新
            pheromone.write(step, decision, confidence)

        pheromone_history.append({
            "step": step,
            "node_pheromones": list(pheromone.values),
            "global_pheromone": pheromone.global_pheromone,
        })

    # Aggregation
    decisions = [o.get("decision", "UNKNOWN") for o in outputs]
    confidences = [float(o.get("confidence", 0.5)) for o in outputs]

    if pheromone_mode == "prompt_only":
        # 条件D: majority_vote を使う（weighted_voteを使わない）
        counts = Counter(d for d in decisions if d != "UNKNOWN")
        final = counts.most_common(1)[0][0] if counts else "UNKNOWN"
    else:
        # dynamic / random / no_prompt: weighted_vote を使う
        final = pheromone.weighted_vote(decisions, confidences)

    return {
        "final_decision": final,
        "decisions": decisions,
        "outputs": outputs,
        "method": "pheromone_weighted",
        "pheromone_history": pheromone_history,
        "final_global_pheromone": pheromone.global_pheromone,
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_experiment(benchmark: str, condition: str, n_samples: int = 50, pheromone_mode: str = "dynamic"):
    models = CONDITIONS[condition]
    use_stigmergy = condition == "stigmergy_nca"

    # Load benchmark
    if benchmark == "alphanli":
        tasks = load_alphanli(n_samples)
    elif benchmark == "ruletaker_d1":
        tasks = load_ruletaker_d1(n_samples)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Output path
    results_dir = REPO_ROOT / "results" / "nca_stigmergy"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{benchmark}_{condition}_{pheromone_mode}.jsonl"

    print(f"[Stigmergy-NCA] benchmark={benchmark} condition={condition}")
    print(f"  models: {models}")
    print(f"  stigmergy: {use_stigmergy}")
    print(f"  pheromone_mode: {pheromone_mode}")
    print(f"  n_samples: {len(tasks)}")
    print(f"  output: {results_path}")

    correct = 0
    total = 0

    with open(results_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            t0 = time.time()

            if use_stigmergy:
                result = run_nca_stigmergy(task, models, pheromone_mode=pheromone_mode)
            else:
                result = run_nca_standard(task, models)

            elapsed = time.time() - t0
            is_correct = result["final_decision"] == label_to_decision(task.label)

            record = {
                "task_id": i,
                "benchmark": benchmark,
                "condition": condition,
                "label": str(task.label),
                "final_decision": result["final_decision"],
                "correct": is_correct,
                "decisions": result["decisions"],
                "method": result["method"],
                "elapsed_sec": round(elapsed, 2),
            }

            # Save pheromone trace only for stigmergy condition
            if use_stigmergy:
                record["pheromone_history"] = result["pheromone_history"]
                record["final_global_pheromone"] = result["final_global_pheromone"]

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            correct += int(is_correct)
            total += 1

            if (i + 1) % 10 == 0:
                acc = correct / total * 100
                print(f"  [{i+1}/{len(tasks)}] acc={acc:.1f}%")

    final_acc = correct / total * 100 if total > 0 else 0.0
    print(f"\n[Done] accuracy={final_acc:.1f}% ({correct}/{total})")
    print(f"Results saved: {results_path}")
    return final_acc


# ── Mirror Effect measurement ─────────────────────────────────────────────────

def measure_mirror_effect(results_path: Path) -> dict:
    """
    副次指標: Mirror Effect の測定。
    全ノードが同じ決定をした割合（unanimity rate）を計算する。
    高い unanimity は mirror effect の可能性を示す。
    """
    records = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        return {}

    total = len(records)
    unanimous = sum(
        1 for r in records
        if len(set(d for d in r["decisions"] if d != "UNKNOWN")) == 1
    )
    # Correct unanimous vs incorrect unanimous
    correct_unanimous = sum(
        1 for r in records
        if len(set(d for d in r["decisions"] if d != "UNKNOWN")) == 1
        and r["correct"]
    )

    mirror_rate = unanimous / total
    # Mirror effect indicator: high unanimity + low accuracy = mirror effect
    accuracy = sum(r["correct"] for r in records) / total

    return {
        "total": total,
        "unanimity_rate": round(mirror_rate, 4),
        "accuracy": round(accuracy, 4),
        "mirror_effect_score": round(mirror_rate * (1 - accuracy), 4),
        # High score = high unanimity but wrong → Mirror Effect
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["alphanli", "ruletaker_d1"],
        required=True,
    )
    parser.add_argument(
        "--condition",
        choices=list(CONDITIONS.keys()),
        default="stigmergy_nca",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--pheromone_mode",
        choices=["dynamic", "random", "no_prompt", "prompt_only"],
        default="dynamic",
        help=(
            "dynamic: normal pheromone update (条件A) / "
            "random: random pheromone signal (条件D相当) / "
            "no_prompt: weighted_vote only, no pheromone prompt (条件C) / "
            "prompt_only: random prompt + majority_vote (条件D)"
        ),
    )
    parser.add_argument(
        "--mirror_only",
        action="store_true",
        help="Only compute mirror effect on existing results (no new inference)",
    )
    args = parser.parse_args()

    if args.mirror_only:
        results_dir = REPO_ROOT / "results" / "nca_stigmergy"
        results_path = results_dir / f"{args.benchmark}_{args.condition}_{args.pheromone_mode}.jsonl"
        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            return
        stats = measure_mirror_effect(results_path)
        print(json.dumps(stats, indent=2))
        return

    run_experiment(args.benchmark, args.condition, args.n_samples, args.pheromone_mode)

    # Auto-compute mirror effect after run
    results_dir = REPO_ROOT / "results" / "nca_stigmergy"
    results_path = results_dir / f"{args.benchmark}_{args.condition}_{args.pheromone_mode}.jsonl"
    if results_path.exists():
        print("\n── Mirror Effect measurement ──")
        stats = measure_mirror_effect(results_path)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
