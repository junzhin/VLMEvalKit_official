"""
MCQ 场景下的 EM / F1 评测脚本

逻辑来源: VLMEvalKit/vlmeval/dataset/medrare_ragbench2.py

两种 prediction 处理方式:
1. prediction 是单个字母 (A-H) → 查找对应选项列的文本内容
2. prediction 是长文本 → 直接使用

然后用解析后的 prediction 与 answer_text + answer_aliases 计算 EM 和 F1。
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
    "output_medrare_20260217/Gemini-2.5-Pro/T20260216_G2c25371d/"
    "Gemini-2.5-Pro_MedRareRAG_Diagnosis.xlsx"
)
DEFAULT_OUTPUT_DIR = ""  # 空则与输入文件同目录
OPTION_COLUMNS = list("ABCDEFGH")  # MCQ 选项列名


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
# Prediction 解析: 字母 → 选项文本 / 长文本 → 直接使用
# ═══════════════════════════════════════════════════════════
def resolve_prediction(row):
    """将 prediction 解析为实际文本。

    如果 prediction 是单个字母 (A-H) 且对应选项列存在且非空，
    则返回该选项列的文本；否则直接返回 prediction 原文。
    """
    pred = str(row.get("prediction", "")).strip()
    if re.match(r"^[A-Ha-h]$", pred):
        col = pred.upper()
        if col in row.index:
            option_text = row[col]
            if pd.notna(option_text) and str(option_text).strip():
                return str(option_text).strip()
    return pred


def build_gold_answers(row):
    """构建 gold answer 列表: answer_text + answer_aliases。"""
    golds = []
    answer_text = row.get("answer_text", "")
    if pd.notna(answer_text) and str(answer_text).strip():
        golds.append(str(answer_text).strip())
    aliases = parse_json_list(row.get("answer_aliases", ""))
    golds.extend([str(a).strip() for a in aliases if str(a).strip()])
    # fallback: 如果都没有，用 answer 列
    if not golds:
        ans = row.get("answer", "")
        if pd.notna(ans) and str(ans).strip():
            golds.append(str(ans).strip())
    return golds


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════
def evaluate(input_file, output_dir=None):
    print(f"读取文件: {input_file}")
    df = pd.read_excel(input_file)
    assert "prediction" in df.columns, "缺少 prediction 列"
    print(f"共 {len(df)} 条数据")

    # 解析 prediction
    df["resolved_prediction"] = df.apply(resolve_prediction, axis=1)

    # 统计字母型 vs 文本型
    letter_mask = df["prediction"].astype(str).str.strip().str.match(r"^[A-Ha-h]$")
    n_letter = letter_mask.sum()
    print(f"字母型 prediction: {n_letter}/{len(df)}, 文本型: {len(df) - n_letter}/{len(df)}")

    # 计算 EM / F1
    em_scores, f1_scores = [], []
    for _, row in df.iterrows():
        pred = row["resolved_prediction"]
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

    import json as _json
    with open(out_summary, "w", encoding="utf-8") as f:
        _json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"汇总结果已保存: {out_summary}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCQ 场景 EM/F1 评测")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE,
                        help="输入 xlsx 文件路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录 (默认与输入文件同目录)")
    args = parser.parse_args()
    evaluate(args.input_file, args.output_dir or None)
