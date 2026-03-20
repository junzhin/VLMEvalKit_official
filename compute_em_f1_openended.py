"""
开放式问答场景下的 EM / F1 评测脚本

逻辑来源: VLMEvalKit/vlmeval/dataset/medrare_ragbench2.py

与 MCQ 版本的区别: prediction 为长文本，直接与 answer 列对比计算 EM/F1。
如存在 gold_answers 列则优先使用，否则用 answer 列。
"""

import argparse
import json
import os
import re
import string
from collections import Counter

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════
# 默认参数 (可在此处直接修改，也可通过命令行覆盖)
# ═══════════════════════════════════════════════════════════
DEFAULT_INPUT_FILE = (
    "/inspire/hdd/global_user/hejunjun-24017/jiyao/Project/20260101_MedQ-Robust/evaluation/VLMEvalKit/"
    "output_medrare_20260217/Claude-Sonnet-4.5/T20260216_G2c25371d/"
    "Claude-Sonnet-4.5_MedRareRAG_Crossmodal.xlsx"
)
DEFAULT_OUTPUT_DIR = ""  # 空则与输入文件同目录


# ═══════════════════════════════════════════════════════════
# EM / F1 核心函数 (from medrare_ragbench2.py:34-65)
# ═══════════════════════════════════════════════════════════
def normalize_answer(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def compute_em(gold_list, predicted):
    return max(
        1.0 if normalize_answer(g) == normalize_answer(predicted) else 0.0
        for g in gold_list
    )


def compute_f1(gold_list, predicted):
    def _f1(gold, pred):
        gold_tokens = normalize_answer(gold).split()
        pred_tokens = normalize_answer(pred).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)
    return max(_f1(g, predicted) for g in gold_list)


def parse_json_list(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == '':
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [s.strip() for s in str(val).split(';') if s.strip()]


# ═══════════════════════════════════════════════════════════
# Gold answers 构建
# ═══════════════════════════════════════════════════════════
def build_gold_answers(row):
    """直接使用 answer 列作为 gold answer。"""
    ans = row.get("answer", "")
    if pd.notna(ans) and str(ans).strip():
        return [str(ans).strip()]
    return []


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════
def evaluate(input_file, output_dir=None):
    print(f"读取文件: {input_file}")
    df = pd.read_excel(input_file)
    assert "prediction" in df.columns, "缺少 prediction 列"
    print(f"共 {len(df)} 条数据")

    df["prediction"] = [str(x) for x in df["prediction"]]

    # 计算 EM / F1
    em_scores, f1_scores = [], []
    for _, row in df.iterrows():
        pred = str(row["prediction"])
        gold_list = build_gold_answers(row)
        if not gold_list:
            em_scores.append(0.0)
            f1_scores.append(0.0)
            continue
        em_scores.append(compute_em(gold_list, pred))
        f1_scores.append(compute_f1(gold_list, pred))

    df["em"] = em_scores
    df["f1"] = f1_scores

    # 汇总
    results = {
        "Overall_EM": np.mean(em_scores) * 100,
        "Overall_F1": np.mean(f1_scores) * 100,
        "n_total": len(df),
    }

    # 按 category 分组
    if "category" in df.columns:
        for cat in sorted(df["category"].dropna().unique()):
            mask = df["category"] == cat
            results[f"{cat}_EM"] = np.mean(df.loc[mask, "em"].values) * 100
            results[f"{cat}_F1"] = np.mean(df.loc[mask, "f1"].values) * 100

    # 按 primary_modality 分组
    if "primary_modality" in df.columns:
        for mod in sorted(df["primary_modality"].dropna().unique()):
            mask = df["primary_modality"] == mod
            results[f"mod_{mod}_EM"] = np.mean(df.loc[mask, "em"].values) * 100
            results[f"mod_{mod}_F1"] = np.mean(df.loc[mask, "f1"].values) * 100

    # 打印结果
    print("\n" + "=" * 50)
    print("评测结果:")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # 保存
    if not output_dir:
        output_dir = os.path.dirname(input_file)
    base = os.path.splitext(os.path.basename(input_file))[0]
    out_detail = os.path.join(output_dir, f"{base}_em_f1_detail.xlsx")
    out_summary = os.path.join(output_dir, f"{base}_em_f1_summary.json")

    df.to_excel(out_detail, index=False)
    print(f"\n详细结果已保存: {out_detail}")

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"汇总结果已保存: {out_summary}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="开放式问答 EM/F1 评测")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE,
                        help="输入 xlsx 文件路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录 (默认与输入文件同目录)")
    args = parser.parse_args()
    evaluate(args.input_file, args.output_dir or None)
