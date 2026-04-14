import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results" / "nca_hgnn"

def load_jsonl(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records

def get_features(r):
    agreements = r.get("agreements", {})
    high_agr = agreements.get("high", 0.0)
    low_agr  = agreements.get("low",  0.0)
    mid_agr  = agreements.get("mid",  None)

    buckets = r.get("buckets", {})
    high_nodes = buckets.get("high", [])
    low_nodes  = buckets.get("low",  [])

    high_decision = _majority([n["decision"] for n in high_nodes if n.get("decision") not in ("UNKNOWN", None)])
    low_decision  = _majority([n["decision"] for n in low_nodes  if n.get("decision") not in ("UNKNOWN", None)])

    all_confs = [n["confidence"] for b in buckets.values() for n in b if n.get("confidence")]
    conf_mean = sum(all_confs) / len(all_confs) if all_confs else 0.5

    agr_diff = high_agr - low_agr
    hl_agree = (high_decision == low_decision) if (high_decision and low_decision) else None

    return {
        "high_agr":   high_agr,
        "low_agr":    low_agr,
        "agr_diff":   agr_diff,
        "hl_agree":   hl_agree,
        "conf_mean":  conf_mean,
        "high_decision": high_decision,
        "low_decision":  low_decision,
    }

def _majority(decisions):
    if not decisions:
        return None
    from collections import Counter
    return Counter(decisions).most_common(1)[0][0]

def bucket_accuracy(records, condition_fn, select_fn):
    correct = 0
    total   = 0
    for r in records:
        label = r.get("label")
        if label is None:
            continue
        true_label = "1" if str(label) in ("True", "1") else "2"
        feat = get_features(r)
        if not condition_fn(feat):
            continue
        selected = select_fn(r, feat)
        if selected == true_label:
            correct += 1
        total += 1
    return correct, total

def analyze(records):
    print(f"Total records: {len(records)}")
    print()

    # ベースライン
    flat_correct = sum(1 for r in records if _get_flat_correct(r))
    print(f"Flat majority baseline: {flat_correct/len(records)*100:.1f}% ({flat_correct}/{len(records)})")
    print()

    # 条件グリッド
    conditions = [
        ("high_agr >= 0.9",
         lambda f: f["high_agr"] >= 0.9),
        ("high_agr >= 0.8",
         lambda f: f["high_agr"] >= 0.8),
        ("high_agr >= 0.7",
         lambda f: f["high_agr"] >= 0.7),
        ("high_agr < 0.7",
         lambda f: f["high_agr"] < 0.7),
        ("agr_diff >= 0.2 (high >> low)",
         lambda f: f["agr_diff"] >= 0.2),
        ("agr_diff <= -0.2 (low >> high)",
         lambda f: f["agr_diff"] <= -0.2),
        ("hl_agree == True",
         lambda f: f["hl_agree"] is True),
        ("hl_agree == False (対立)",
         lambda f: f["hl_agree"] is False),
        ("conf_mean >= 0.85",
         lambda f: f["conf_mean"] >= 0.85),
        ("conf_mean < 0.7",
         lambda f: f["conf_mean"] < 0.7),
    ]

    selectors = [
        ("high選択",  lambda r, f: f["high_decision"] or r.get("flat_decision")),
        ("low選択",   lambda r, f: f["low_decision"]  or r.get("flat_decision")),
        ("flat選択",  lambda r, f: r.get("flat_decision")),
    ]

    print("=" * 70)
    print(f"{'条件':<35} {'選択':<10} {'正解率':>8} {'件数':>6}")
    print("=" * 70)

    best_rules = []

    for cond_name, cond_fn in conditions:
        for sel_name, sel_fn in selectors:
            correct, total = bucket_accuracy(records, cond_fn, sel_fn)
            if total == 0:
                continue
            acc = correct / total * 100
            flat_c, flat_t = bucket_accuracy(records, cond_fn,
                lambda r, f: r.get("flat_decision"))
            flat_acc = flat_c / flat_t * 100 if flat_t > 0 else 0
            diff = acc - flat_acc
            marker = " ★" if diff >= 3.0 else ""
            print(f"{cond_name:<35} {sel_name:<10} {acc:>7.1f}% {total:>5}{marker}")

            if diff >= 3.0:
                best_rules.append({
                    "condition": cond_name,
                    "selector":  sel_name,
                    "accuracy":  round(acc, 1),
                    "n":         total,
                    "diff_vs_flat": round(diff, 1),
                })
        print()

    print()
    print("=" * 70)
    print("有効ルール候補 (flat比+3pp以上):")
    print("=" * 70)
    if best_rules:
        for rule in sorted(best_rules, key=lambda x: -x["diff_vs_flat"]):
            print(f"  IF {rule['condition']}")
            print(f"    THEN {rule['selector']}")
            print(f"    → {rule['accuracy']}% (n={rule['n']}, flat比{rule['diff_vs_flat']:+.1f}pp)")
            print()
    else:
        print("  flat比+3pp以上のルールは見つかりませんでした。")

    return best_rules

def _get_flat_correct(r):
    label = r.get("label")
    if label is None:
        return False
    true_label = "1" if str(label) in ("True", "1") else "2"
    return r.get("flat_decision") == true_label

def generate_hyperedge_code(rules):
    if not rules:
        return None

    lines = []
    lines.append("# 自動生成された超エッジルール")
    lines.append("# analyze_hyperedge.py により生成")
    lines.append("def hyperedge_select(feat, high_decision, low_decision, flat_decision):")
    lines.append('    """')
    lines.append("    自動生成超エッジ: 条件に応じてbucketを選択する。")
    lines.append('    """')

    for rule in sorted(rules, key=lambda x: -x["diff_vs_flat"]):
        cond = rule["condition"]
        sel  = rule["selector"]

        if "high_agr >= 0.9" in cond:
            py_cond = "feat['high_agr'] >= 0.9"
        elif "high_agr >= 0.8" in cond:
            py_cond = "feat['high_agr'] >= 0.8"
        elif "high_agr >= 0.7" in cond:
            py_cond = "feat['high_agr'] >= 0.7"
        elif "high_agr < 0.7" in cond:
            py_cond = "feat['high_agr'] < 0.7"
        elif "agr_diff >= 0.2" in cond:
            py_cond = "feat['agr_diff'] >= 0.2"
        elif "agr_diff <= -0.2" in cond:
            py_cond = "feat['agr_diff'] <= -0.2"
        elif "hl_agree == True" in cond:
            py_cond = "feat['hl_agree'] is True"
        elif "hl_agree == False" in cond:
            py_cond = "feat['hl_agree'] is False"
        elif "conf_mean >= 0.85" in cond:
            py_cond = "feat['conf_mean'] >= 0.85"
        elif "conf_mean < 0.7" in cond:
            py_cond = "feat['conf_mean'] < 0.7"
        else:
            continue

        if "high" in sel:
            ret = "high_decision or flat_decision"
        elif "low" in sel:
            ret = "low_decision or flat_decision"
        else:
            ret = "flat_decision"

        lines.append(f"    if {py_cond}:  # {rule['accuracy']}% n={rule['n']} flat比{rule['diff_vs_flat']:+.1f}pp")
        lines.append(f"        return {ret}")

    lines.append("    return flat_decision  # fallback")
    return "\n".join(lines)

def main():
    files = list(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        print(f"jsonlファイルが見つかりません: {RESULTS_DIR}")
        return

    all_records = []
    for f in files:
        recs = load_jsonl(f)
        print(f"Loaded {len(recs)} records from {f.name}")
        all_records.extend(recs)

    print()
    rules = analyze(all_records)

    code = generate_hyperedge_code(rules)
    if code:
        out_path = REPO_ROOT / "hyperedge_rules.py"
        out_path.write_text(code, encoding="utf-8")
        print(f"\n超エッジコード生成: {out_path}")
        print()
        print(code)
    else:
        print("\n有効なルールが見つからなかったため、コード生成をスキップしました。")

if __name__ == "__main__":
    main()
