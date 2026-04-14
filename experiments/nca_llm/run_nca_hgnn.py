"""
run_nca_hgnn.py

HGNN-bucket実験 Stage 1: 5ノード × 2分割。
全ノード完全非通信（contextなし）で独立推論。
confidence分位数で low/high に2分割し、
agreement率が高いバケットの多数決を最終回答とする。

検証仮説:
  仮説X: high-confidenceバケットのagreementが高い問題では
         high群の多数決の正解率が高い
  仮説Y: low-confidenceバケットのagreementが高い問題では
         low群の多数決の方が正解率が高い

Usage:
  python run_nca_hgnn.py --benchmark alphanli --n_samples 50
  python run_nca_hgnn.py --benchmark ruletaker_d1 --n_samples 50
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

# benchmark_loader.py は nca-llm-external-benchmark リポジトリに存在
_NCA_EXT = Path(__file__).resolve().parent.parent.parent.parent / "nca-llm-external-benchmark" / "experiments" / "nca_llm"
if _NCA_EXT.exists():
    sys.path.insert(0, str(_NCA_EXT))

from benchmark_loader import load_alphanli, load_ruletaker_d1

OLLAMA_URL = "http://localhost:11434/api/generate"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODELS_5NODE = [
    "qwen2.5:7b",
    "qwen2.5:7b",
    "llama3:latest",
    "mistral:7b",
    "gemma2:9b",
]

_client = httpx.Client(timeout=120.0)


# ─────────────────────────────────────────────
#  LLM 呼び出し
# ─────────────────────────────────────────────

def call_llm(model: str, prompt: str) -> dict:
    """Ollama /api/generate を呼び出し、JSON応答をパースして返す。"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = _client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        text = resp.json().get("response", "")
    except Exception as e:
        print(f"  [LLM error] {model}: {e}")
        return {"decision": "UNKNOWN", "confidence": 0.0, "reasoning": str(e)}

    # JSON部分を抽出してパース
    try:
        # レスポンス全体がJSONの場合
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # JSON部分を抽出
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    return {
        "decision": str(parsed.get("decision", "UNKNOWN")),
        "confidence": float(parsed.get("confidence", 0.5)),
        "reasoning": parsed.get("reasoning", text),
    }


# ─────────────────────────────────────────────
#  プロンプト構築
# ─────────────────────────────────────────────

def build_base_prompt(task) -> str:
    benchmark = task.benchmark
    if benchmark == "alphanli":
        return (
            f"{task.question}\n\n"
            "Answer 1: Yes, the hypothesis is correct.\n"
            "Answer 2: No, the hypothesis is not correct.\n"
            'Respond in JSON: {"decision": "1" or "2", '
            '"confidence": 0.0-1.0, "reasoning": "..."}'
        )
    elif benchmark == "ruletaker_d1":
        return (
            f"{task.question}\n\n"
            "Answer 1: TRUE\n"
            "Answer 2: FALSE\n"
            'Respond in JSON: {"decision": "1" (TRUE) or "2" (FALSE), '
            '"confidence": 0.0-1.0, "reasoning": "..."}'
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def label_to_decision(label) -> str:
    if label is True or label == "True" or label == 1:
        return "1"
    elif label is False or label == "False" or label == 0:
        return "2"
    return str(label)


# ─────────────────────────────────────────────
#  HGNN-bucket コア関数
# ─────────────────────────────────────────────

def quantile_split(outputs: list[dict]) -> dict:
    """
    confidence の分位数で outputs を low / high に2分割する。
    n=5の場合: low=下位2ノード、high=上位3ノード
    """
    sorted_outputs = sorted(
        outputs,
        key=lambda o: float(o.get("confidence", 0.5))
    )
    cut = len(sorted_outputs) // 2
    return {
        "low":  sorted_outputs[:cut],
        "high": sorted_outputs[cut:],
    }


def bucket_agreement(bucket: list[dict]) -> float:
    """
    バケット内のノードが同じdecisionに投票している割合。
    1ノード以下の場合は 0.0 を返す（信頼しない）。
    """
    if len(bucket) < 2:
        return 0.0
    decisions = [o.get("decision", "UNKNOWN") for o in bucket
                 if o.get("decision") not in ("UNKNOWN", None)]
    if not decisions:
        return 0.0
    most_common_count = Counter(decisions).most_common(1)[0][1]
    return most_common_count / len(decisions)


def majority_vote_bucket(bucket: list[dict]) -> str:
    """バケット内のmajority_vote。"""
    decisions = [o.get("decision", "UNKNOWN") for o in bucket
                 if o.get("decision") not in ("UNKNOWN", None)]
    if not decisions:
        return "UNKNOWN"
    return Counter(decisions).most_common(1)[0][0]


# ─────────────────────────────────────────────
#  メイン推論パイプライン
# ─────────────────────────────────────────────

def run_hgnn_2split(task, models: list[str]) -> dict:
    """
    全ノード独立推論 → 2分割 → agreement率最大バケットで多数決。
    """
    # Step 1: 全ノード独立推論（contextなし）
    outputs = []
    for model in models:
        prompt = build_base_prompt(task)
        out = call_llm(model, prompt)
        outputs.append({"model": model, **out})

    # Step 2: 分位数で2分割
    buckets = quantile_split(outputs)

    # Step 3: 各バケットのagreement率を計算
    agreements = {
        k: bucket_agreement(v)
        for k, v in buckets.items()
    }

    # Step 4: agreement率が最大のバケットを選択
    # 同率の場合はhighを優先
    best_bucket_key = max(
        agreements,
        key=lambda k: (agreements[k], 1 if k == "high" else 0)
    )
    best_bucket = buckets[best_bucket_key]

    # Step 5: 選択バケット内でmajority_vote
    final = majority_vote_bucket(best_bucket)

    # 全ノードflatのmajority_voteも記録（baseline比較用）
    all_decisions = [o.get("decision", "UNKNOWN") for o in outputs]
    flat_final = majority_vote_bucket(outputs)

    return {
        "final_decision": final,
        "flat_decision": flat_final,
        "selected_bucket": best_bucket_key,
        "agreements": agreements,
        "buckets": {
            k: [{"model": o["model"],
                 "decision": o.get("decision"),
                 "confidence": o.get("confidence")}
                for o in v]
            for k, v in buckets.items()
        },
        "all_decisions": all_decisions,
        "outputs": outputs,
    }


# ─────────────────────────────────────────────
#  実験実行
# ─────────────────────────────────────────────

def run_experiment(benchmark: str, n_samples: int = 50):
    models = MODELS_5NODE

    if benchmark == "alphanli":
        tasks = load_alphanli(n_samples)
    elif benchmark == "ruletaker_d1":
        tasks = load_ruletaker_d1(n_samples)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    results_dir = REPO_ROOT / "results" / "nca_hgnn"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{benchmark}_hgnn_2split.jsonl"

    print(f"[HGNN-2split] benchmark={benchmark}")
    print(f"  models: {models}")
    print(f"  n_samples: {len(tasks)}")
    print(f"  output: {results_path}")

    correct_hgnn = 0
    correct_flat = 0
    total = 0

    # bucket選択の統計
    bucket_selection_counts = Counter()

    with open(results_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            t0 = time.time()

            result = run_hgnn_2split(task, models)
            elapsed = time.time() - t0

            label = label_to_decision(task.label)
            is_correct_hgnn = result["final_decision"] == label
            is_correct_flat = result["flat_decision"] == label

            bucket_selection_counts[result["selected_bucket"]] += 1

            record = {
                "task_id": i,
                "benchmark": benchmark,
                "label": str(task.label),
                "final_decision": result["final_decision"],
                "flat_decision": result["flat_decision"],
                "correct_hgnn": is_correct_hgnn,
                "correct_flat": is_correct_flat,
                "selected_bucket": result["selected_bucket"],
                "agreements": result["agreements"],
                "buckets": result["buckets"],
                "all_decisions": result["all_decisions"],
                "elapsed_sec": round(elapsed, 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            correct_hgnn += int(is_correct_hgnn)
            correct_flat += int(is_correct_flat)
            total += 1

            if (i + 1) % 10 == 0:
                acc_hgnn = correct_hgnn / total * 100
                acc_flat = correct_flat / total * 100
                print(f"  [{i+1}/{len(tasks)}] "
                      f"hgnn={acc_hgnn:.1f}% flat={acc_flat:.1f}%")

    acc_hgnn = correct_hgnn / total * 100 if total > 0 else 0.0
    acc_flat = correct_flat / total * 100 if total > 0 else 0.0

    print(f"\n[Done]")
    print(f"  HGNN-2split : {acc_hgnn:.1f}% ({correct_hgnn}/{total})")
    print(f"  Flat majority: {acc_flat:.1f}% ({correct_flat}/{total})")
    print(f"  Bucket selection: {dict(bucket_selection_counts)}")
    print(f"  Results saved: {results_path}")

    return acc_hgnn, acc_flat


# ─────────────────────────────────────────────
#  エントリーポイント
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["alphanli", "ruletaker_d1"],
        default="alphanli",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
    )
    args = parser.parse_args()
    run_experiment(args.benchmark, args.n_samples)


if __name__ == "__main__":
    main()
