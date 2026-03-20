#!/bin/bash
# ================================================================
# MedRareRAGBench 评测脚本
#
# 用法: bash scripts/medrare_api/run_medrare_ragbench.sh scripts/medrare_api/batch1/xxx.json
# ================================================================

set -e

WORK_DIR="/inspire/hdd/global_user/hejunjun-24017/jiyao/Project/20260101_MedQ-Robust/evaluation/VLMEvalKit"
cd "$WORK_DIR"

# ================================================================
# 环境初始化
# ================================================================
source /inspire/hdd/global_user/hejunjun-24017/jiyao/.bashrc
conda activate vlmeval

# ================================================================
# 配置文件（默认使用传入的参数，否则用 full 配置）
# ================================================================
CONFIG_FILE="${1:-$WORK_DIR/scripts/medrare_api/batch1/medrare_ragbench_release_test.json}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# ================================================================
# 参数设定（直接在此修改）
# ================================================================
OUTPUT_DIR="output"
JUDGE_MODEL="qwen3vl"
API_NPROC=64
CLEAN_CACHE=0
DISABLE_BERTSCORE=1

# Judge API（本地部署的 Qwen3-235B）
JUDGE_API_BASE="https://cckqhcd9ddbkcmacjqohhmdd9co5goa5.openapi-qb.sii.edu.cn/v1/chat/completions"
JUDGE_API_KEY="ssfCnf/XLYOAUiwYshoHn8Ss/VX+bEJaUEtUci3/Qmk="

# 数据路径
LMUDATA="/inspire/hdd/global_user/hejunjun-24017/jiyao/Project/20260101_MedQ-Robust/evaluation/LMUData"

# ================================================================
# 导出环境变量（下方代码依赖）
# ================================================================
export OPENAI_API_BASE="$JUDGE_API_BASE"
export OPENAI_API_KEY="$JUDGE_API_KEY"
export LMUData="$LMUDATA"
export DISABLE_BERTSCORE="$DISABLE_BERTSCORE"

mkdir -p "$OUTPUT_DIR"

# ================================================================
# 从 JSON 配置中提取模型名称列表
# ================================================================
MODEL_NAMES=$(python -c "
import json, sys
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
models = [k for k in cfg.get('model', {}).keys() if not k.startswith('_')]
print('\n'.join(models))
")
MODEL_COUNT=$(echo "$MODEL_NAMES" | wc -l)

# ================================================================
# 打印配置
# ================================================================
echo "=========================================="
CONFIG_LABEL="$(basename "$CONFIG_FILE" .json)"
echo " MedRareRAGBench 评测 [$CONFIG_LABEL]"
echo "=========================================="
echo ""
echo " 配置文件:  $CONFIG_FILE"
echo " 输出目录:  $OUTPUT_DIR"
echo " Judge:     $JUDGE_MODEL (via OPENAI_API_BASE)"
echo " 并发数:    $API_NPROC"
echo " 清缓存:    $CLEAN_CACHE (CLEAN_CACHE=1 时重测前删旧 xlsx/pkl)"
echo ""
echo " 推理模型 ($MODEL_COUNT 个):"
i=1
echo "$MODEL_NAMES" | while read -r m; do
    echo "   $i. $m"
    i=$((i+1))
done
echo ""
echo "=========================================="
echo ""

# ================================================================
# Step 1: 清理旧缓存（CLEAN_CACHE=1 时生效）
# ================================================================
if [ "$CLEAN_CACHE" = "1" ]; then
    echo "[Step 0] 清理旧缓存（CLEAN_CACHE=1）..."
    echo "$MODEL_NAMES" | while read -r model; do
        MODEL_DIR="$OUTPUT_DIR/$model"
        if [ -d "$MODEL_DIR" ]; then
            echo "  清理 $model 的旧推理缓存..."
            rm -rf "$MODEL_DIR"
            echo "  ✓ $model 缓存已清理"
        fi
    done
    echo ""
fi

# ================================================================
# Step 2: 推理 + 评估
# ================================================================
echo "[Step 1] 运行推理和评估..."
echo ""

python run.py \
  --config "$CONFIG_FILE" \
  --judge "$JUDGE_MODEL" \
  --work-dir "$OUTPUT_DIR" \
  --api-nproc "$API_NPROC" \
  --reuse \
  --verbose

# ================================================================
# Step 3: 汇总输出
# ================================================================
echo ""
echo "[Step 2] 生成的文件:"
echo "------------------------------------------"
find "$OUTPUT_DIR" -type f \( -name "*.xlsx" -o -name "*.csv" -o -name "*.json" \) | sort
echo ""

# ================================================================
# Step 4: 聚合 JSON
# ================================================================
echo "[Step 3] 生成汇总 JSON..."
echo ""

if [ -f "scripts/aggregate_medrare_results.py" ]; then
    echo "$MODEL_NAMES" | while read -r model; do
        if ! find "$OUTPUT_DIR" -name "${model}_aggregated_results.json" 2>/dev/null | grep -q .; then
            echo "  聚合 $model ..."
            python scripts/aggregate_medrare_results.py \
              --output_dir "$OUTPUT_DIR" \
              --model_name "$model" 2>/dev/null || echo "  [warn] $model 聚合失败（可能部分 track 未完成）"
        else
            echo "  $model 汇总已存在，跳过"
        fi
    done
fi

# ================================================================
# 结果展示
# ================================================================
echo ""
echo "=========================================="
echo " 评测结果 [$CONFIG_LABEL]"
echo "=========================================="

echo "$MODEL_NAMES" | while read -r model; do
    JSON_FILE=$(find "$OUTPUT_DIR" -name "${model}_aggregated_results.json" 2>/dev/null | head -1)
    if [ -n "$JSON_FILE" ] && [ -f "$JSON_FILE" ]; then
        echo ""
        echo "--- $model ---"
        cat "$JSON_FILE" | python -m json.tool
    fi
done

echo ""
echo "=========================================="
echo " 完成！配置: $CONFIG_LABEL | 输出目录: $OUTPUT_DIR"
echo "=========================================="
