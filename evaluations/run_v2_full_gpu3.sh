#!/bin/bash
# Eval samples 10..27 (18 samples) for TWO models on GPU 3:
#   - lagrpo_all_equal_760
#   - lagrpo_fair_only_760
# Samples 0..9 are already in the existing <name>.json files.
# This pod handles 2 models (vs 3 on the others) so it finishes earlier
# -- buffer for LA-GRPO fair-only's longer dialogues.
#
# Per model: 18 samples x 20 reps = 360 games.
# Per pod: 2 models x 360 = 720 games.
# Outputs: evaluations/results/negotiation/v2/<name>_samples10-27.json
set -e

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="output"
RESULTS_DIR="evaluations/results/negotiation/v2"
REPS=20
GAME_TYPE="multi-game"
START_IDX=10
NUM_GAMES=18
SLICE_TAG="samples${START_IDX}-$((START_IDX + NUM_GAMES - 1))"

run_eval_slice() {
    local ckpt_path="$1"
    local out_name="$2"
    local lam_self="$3"
    local lam_welfare="$4"
    local lam_fair="$5"

    local target="$RESULTS_DIR/${out_name}_${SLICE_TAG}.json"
    if [ -f "$target" ]; then
        echo "[SKIP] $target already exists"
        return 0
    fi

    echo ""
    echo ">>> [GPU $CUDA_VISIBLE_DEVICES] ${out_name}: samples ${START_IDX}..$((START_IDX+NUM_GAMES-1))"
    python evaluations/run_negotiation_eval.py \
        --checkpoint "${ckpt_path}" \
        --repetitions ${REPS} \
        --game-type ${GAME_TYPE} \
        --num-games ${NUM_GAMES} \
        --sample-start-idx ${START_IDX} \
        --output-dir "$RESULTS_DIR" \
        --lambda-self ${lam_self} \
        --lambda-welfare ${lam_welfare} \
        --lambda-fair ${lam_fair}

    local base
    if [ "$ckpt_path" = "none" ]; then
        base="none"
    else
        base=$(basename "${ckpt_path%/}")
    fi
    local src="$RESULTS_DIR/${base}_${SLICE_TAG}.json"
    if [ -f "$src" ] && [ "$src" != "$target" ]; then
        mv "$src" "$target"
    fi
    echo "[DONE] $target"
}

run_eval_slice "output/lagrpo-multigame-all-equal/checkpoint-760" "lagrpo_all_equal_760" 0.33 0.33 0.33
run_eval_slice "output/lagrpo-fair-only/checkpoint-760"           "lagrpo_fair_only_760" 0.0  0.0  1.0

echo ""
echo "============================================"
echo "GPU 3 complete: 2 models x 18 samples x 20 reps = 720 games."
echo "============================================"
ls -la "$RESULTS_DIR"/{lagrpo_all_equal_760,lagrpo_fair_only_760}_${SLICE_TAG}.json 2>/dev/null
