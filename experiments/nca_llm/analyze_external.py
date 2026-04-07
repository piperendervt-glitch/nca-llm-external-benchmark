"""
Analysis script for MVE-20260405-02.
Computes metrics for all 8 conditions and compares with Claude WC reference.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "nca_llm"

BENCHMARKS = ["alphanli", "ruletaker_d1"]
CONDITIONS = ["homo_nca", "het_nca_v1", "het_nca_v2", "majority_vote"]


def load_results(benchmark: str, condition: str) -> list[dict]:
    path = RESULTS_DIR / benchmark / f"{condition}.jsonl"
    if not path.exists():
        return []
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {"n": 0, "acc": 0, "cfr": 0, "uni_pct": 0, "n_split": 0, "split_acc": 0, "cal_err": 0}

    n = len(results)
    n_correct = sum(1 for r in results if r.get("is_correct", False))
    acc = n_correct / n * 100

    # CFR = correct flip rate (incorrect on unanimous, correct on split)
    unanimous = [r for r in results if r.get("is_unanimous", False)]
    split = [r for r in results if not r.get("is_unanimous", False)]
    n_split = len(split)
    uni_pct = len(unanimous) / n * 100

    split_correct = sum(1 for r in split if r.get("is_correct", False))
    split_acc = split_correct / n_split * 100 if n_split > 0 else 0

    # CFR: fraction of split tasks that are correct
    cfr = split_correct / n_split * 100 if n_split > 0 else 0

    # Calibration error (average |confidence - accuracy| per confidence bin)
    bins = {}
    for r in results:
        node_outputs = r.get("node_outputs", {})
        for key, val in node_outputs.items():
            if isinstance(val, dict):
                conf = val.get("confidence", 0.5)
                dec = val.get("decision", "UNKNOWN")
                label = r.get("label", False)
                if dec == "CORRECT":
                    correct = label is True
                elif dec == "INCORRECT":
                    correct = label is False
                else:
                    continue
                bin_key = round(conf, 1)
                if bin_key not in bins:
                    bins[bin_key] = []
                bins[bin_key].append(1 if correct else 0)

    cal_err = 0
    if bins:
        total_err = 0
        total_n = 0
        for conf_bin, outcomes in bins.items():
            bin_acc = sum(outcomes) / len(outcomes)
            total_err += abs(conf_bin - bin_acc) * len(outcomes)
            total_n += len(outcomes)
        cal_err = total_err / total_n * 100 if total_n > 0 else 0

    return {
        "n": n,
        "acc": acc,
        "cfr": cfr,
        "uni_pct": uni_pct,
        "n_split": n_split,
        "split_acc": split_acc,
        "cal_err": cal_err,
    }


def main():
    print("=" * 70)
    print("MVE-20260405-02: External Benchmark Verification")
    print("=" * 70)

    all_metrics = {}

    for benchmark in BENCHMARKS:
        bm_label = "αNLI" if benchmark == "alphanli" else "RuleTaker-depth-1"
        print(f"\nBenchmark: {bm_label}")
        print(f"{'Condition':<20} | {'acc':>5} | {'CFR':>5} | {'uni%':>5} | {'n_spl':>5} | {'spl_acc':>7} | {'cal_err':>7}")
        print("-" * 70)

        for condition in CONDITIONS:
            results = load_results(benchmark, condition)
            metrics = compute_metrics(results)
            all_metrics[(benchmark, condition)] = metrics

            if metrics["n"] == 0:
                print(f"{condition:<20} | {'N/A':>5} | {'N/A':>5} | {'N/A':>5} | {'N/A':>5} | {'N/A':>7} | {'N/A':>7}")
            else:
                print(f"{condition:<20} | {metrics['acc']:>5.1f} | {metrics['cfr']:>5.1f} | {metrics['uni_pct']:>5.1f} | {metrics['n_split']:>5} | {metrics['split_acc']:>7.1f} | {metrics['cal_err']:>7.1f}")

    # CFR comparison
    print("\n" + "=" * 70)
    print("CFR差の比較:")
    print(f"{'':>18} | {'αNLI':>10} | {'RuleTaker-d1':>12} | {'Claude WC(ref)':>14}")
    print("-" * 60)

    comparisons = [
        ("homo vs het_v1", "het_nca_v1", "+16.3pp"),
        ("homo vs het_v2", "het_nca_v2", "—"),
        ("homo vs mv",     "majority_vote", "—"),
    ]

    for label, cond, ref in comparisons:
        vals = []
        for bm in BENCHMARKS:
            homo = all_metrics.get((bm, "homo_nca"), {})
            other = all_metrics.get((bm, cond), {})
            if homo.get("n", 0) > 0 and other.get("n", 0) > 0:
                diff = other["cfr"] - homo["cfr"]
                vals.append(f"{diff:>+.1f}pp")
            else:
                vals.append("N/A")
        print(f"{label:>18} | {vals[0]:>10} | {vals[1]:>12} | {ref:>14}")

    # Success criteria
    print("\n" + "=" * 70)
    print("Success Criteria（事前宣言・変更禁止）:")

    successes = 0
    total_criteria = 0

    for het_cond in ["het_nca_v1", "het_nca_v2"]:
        print(f"\n  CFR差(homo-{het_cond}) > +5pp:")
        for bm in BENCHMARKS:
            bm_label = "αNLI" if bm == "alphanli" else "RuleTaker-d1"
            homo = all_metrics.get((bm, "homo_nca"), {})
            het = all_metrics.get((bm, het_cond), {})
            total_criteria += 1
            if homo.get("n", 0) > 0 and het.get("n", 0) > 0:
                diff = het["cfr"] - homo["cfr"]
                if diff > 5.0:
                    result = "SUCCESS"
                    successes += 1
                else:
                    result = "FAIL"
                print(f"    {bm_label}: {result} ({diff:+.1f}pp)")
            else:
                print(f"    {bm_label}: NO DATA")

    print()
    if successes == total_criteria:
        judgment = "FULL SUCCESS"
    elif successes > 0:
        judgment = "PARTIAL"
    else:
        judgment = "FAIL"
    print(f"判定: {judgment} ({successes}/{total_criteria})")
    print("=" * 70)


if __name__ == "__main__":
    main()
