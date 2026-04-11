"""
compare_stigmergy.py

Stigmergy-NCA vs existing conditions の比較分析。

既存条件のキー: is_correct, is_unanimous, node_outputs
Stigmergy条件のキー: correct, decisions

Usage:
  python compare_stigmergy.py --benchmark alphanli
  python compare_stigmergy.py --benchmark ruletaker_d1
  python compare_stigmergy.py --benchmark alphanli --plot
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXISTING_CONDITIONS = ["homo_nca", "het_nca_v1", "het_nca_v2", "majority_vote"]


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_stigmergy(benchmark):
    path = REPO_ROOT / "results" / "nca_stigmergy" / f"{benchmark}_stigmergy_nca.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Stigmergy results not found: {path}")
    return load_jsonl(path)


def load_existing(benchmark, condition):
    path = REPO_ROOT / "results" / "nca_llm" / benchmark / f"{condition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return load_jsonl(path)


def get_correct(r):
    if "is_correct" in r:
        return bool(r["is_correct"])
    return bool(r.get("correct", False))


def get_is_unanimous(r):
    if "is_unanimous" in r:
        return bool(r["is_unanimous"])
    decisions = [d for d in r.get("decisions", []) if d not in ("UNKNOWN", None, "")]
    return len(decisions) > 0 and len(set(decisions)) == 1


def compute_metrics(records, n_samples=None):
    if n_samples:
        records = records[:n_samples]
    total = len(records)
    if total == 0:
        return {}

    correct = sum(1 for r in records if get_correct(r))
    accuracy = correct / total

    unanimous_wrong = 0
    unanimous_correct = 0
    for r in records:
        if get_is_unanimous(r):
            if get_correct(r):
                unanimous_correct += 1
            else:
                unanimous_wrong += 1

    unanimity_rate = (unanimous_correct + unanimous_wrong) / total
    mirror_effect_score = unanimous_wrong / total

    return {
        "n": total,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 1),
        "unanimity_rate": round(unanimity_rate, 4),
        "mirror_effect_score": round(mirror_effect_score, 4),
    }


def compute_stigmergy_extras(records):
    values = [r["final_global_pheromone"] for r in records if "final_global_pheromone" in r]
    if not values:
        return {}
    neutral = sum(1 for v in values if 0.4 <= v <= 0.6)
    decisive = sum(1 for v in values if v < 0.3 or v > 0.7)
    return {
        "pheromone_mean": round(sum(values) / len(values), 4),
        "pheromone_neutral_rate": round(neutral / len(values), 4),
        "pheromone_decisive_rate": round(decisive / len(values), 4),
    }


def compare(benchmark, plot=False):
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark}")
    print(f"{'='*60}")

    try:
        stigmergy_records = load_stigmergy(benchmark)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    n = len(stigmergy_records)
    print(f"Stigmergy samples: {n}")
    print(f"Existing conditions compared using first {n} samples.\n")

    results = {}
    results["stigmergy_nca"] = {
        **compute_metrics(stigmergy_records),
        **compute_stigmergy_extras(stigmergy_records),
    }

    for cond in EXISTING_CONDITIONS:
        try:
            records = load_existing(benchmark, cond)
            results[cond] = compute_metrics(records, n_samples=n)
        except FileNotFoundError:
            print(f"  WARNING: {cond} not found, skipping")

    # Table
    print(f"{'Condition':<20} {'Accuracy':>10} {'Unanimity':>12} {'MirrorEffect':>14}")
    print("-" * 60)
    for cond, m in results.items():
        marker = " ★" if cond == "stigmergy_nca" else ""
        print(
            f"{cond:<20} "
            f"{m['accuracy_pct']:>9.1f}% "
            f"{m['unanimity_rate']:>11.3f} "
            f"{m['mirror_effect_score']:>13.3f}"
            f"{marker}"
        )

    # vs het_nca_v1
    if "het_nca_v1" in results and "stigmergy_nca" in results:
        b = results["het_nca_v1"]
        s = results["stigmergy_nca"]
        acc_diff = s["accuracy_pct"] - b["accuracy_pct"]
        mirror_diff = s["mirror_effect_score"] - b["mirror_effect_score"]
        print(f"\n── Stigmergy vs het_nca_v1 (same models) ──")
        print(f"  Accuracy diff     : {acc_diff:+.1f}pp")
        print(f"  Mirror Effect diff: {mirror_diff:+.4f}")
        if acc_diff > 2:
            print(f"  → Stigmergy improved accuracy by {acc_diff:.1f}pp")
        elif acc_diff < -2:
            print(f"  → Stigmergy hurt accuracy by {abs(acc_diff):.1f}pp")
        else:
            print(f"  → Accuracy difference within noise range (±2pp)")
        if mirror_diff < -0.05:
            print(f"  → Mirror Effect reduced ✓")
        elif mirror_diff > 0.05:
            print(f"  → Mirror Effect increased")
        else:
            print(f"  → Mirror Effect unchanged")

    # Pheromone
    s = results.get("stigmergy_nca", {})
    if "pheromone_mean" in s:
        print(f"\n── Pheromone analysis ──")
        print(f"  Mean global pheromone        : {s['pheromone_mean']:.4f}")
        print(f"  Neutral rate  (0.4–0.6)      : {s['pheromone_neutral_rate']:.3f}")
        print(f"  Decisive rate (<0.3 or >0.7) : {s['pheromone_decisive_rate']:.3f}")
        if s["pheromone_neutral_rate"] > 0.8:
            print(f"  ⚠ フェロモンが中立に集中 → 信号として機能していない可能性")

    # Save
    summary_path = REPO_ROOT / "results" / "nca_stigmergy" / f"{benchmark}_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    # Plot
    if plot:
        try:
            import matplotlib.pyplot as plt
            conditions = list(results.keys())
            accuracies = [results[c]["accuracy_pct"] for c in conditions]
            mirror_scores = [results[c]["mirror_effect_score"] for c in conditions]
            colors = ["#E85D30" if c == "stigmergy_nca" else "#888780" for c in conditions]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].bar(conditions, accuracies, color=colors)
            axes[0].set_title(f"Accuracy — {benchmark} (n={n})")
            axes[0].set_ylabel("Accuracy (%)")
            axes[0].set_ylim(0, 100)
            for i, v in enumerate(accuracies):
                axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
            axes[0].tick_params(axis="x", rotation=30)

            axes[1].bar(conditions, mirror_scores, color=colors)
            axes[1].set_title(f"Mirror Effect Score — {benchmark} (n={n})")
            axes[1].set_ylabel("unanimous & wrong rate")
            axes[1].set_ylim(0, max(mirror_scores) * 1.4 + 0.01)
            for i, v in enumerate(mirror_scores):
                axes[1].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
            axes[1].tick_params(axis="x", rotation=30)

            plt.tight_layout()
            plot_path = REPO_ROOT / "results" / "nca_stigmergy" / f"{benchmark}_comparison.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {plot_path}")
            plt.show()
        except ImportError:
            print("matplotlib not installed — pip install matplotlib")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["alphanli", "ruletaker_d1", "all"], default="alphanli")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.benchmark == "all":
        compare("alphanli", args.plot)
        compare("ruletaker_d1", args.plot)
    else:
        compare(args.benchmark, args.plot)


if __name__ == "__main__":
    main()
