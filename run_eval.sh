#!/bin/bash
#SBATCH --job-name=dmn_eval
#SBATCH --account=ACD109125
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# 四個模型的 MMD 與 KS。四個目錄都要有 test_graphs.pkl 與 sampled_ts.pkl，
# 缺任何一個都會中斷（temporal_baseline_evaluator.py 直接開檔，沒有容錯）。
#
# walltime 設 48 小時（與 run_damnets_pipeline.sh 相同）。MMD 是 O(序列數²)，
# twitter 與 superuser 的計算量約為 wiki-vote 的 60 至 160 倍，
# 原本的 8 小時不夠，單獨補跑會 TIMEOUT。
#
#   sbatch run_eval.sh wiki-vote 123

set -eo pipefail
module load miniconda3/conda24.5.0_py3.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate damnets
export PYTHONNOUSERSITE=1

# python 的 stdout 接到管線（下面的 tee）時是區塊緩衝的，進度會卡在緩衝區裡，
# 從 log 完全看不出跑到哪。關掉緩衝才看得到即時進度。
export PYTHONUNBUFFERED=1

# MMD 是 O(序列數²)：dist_helper.disc() 對兩組圖做所有配對的 EMD，
# compute_mmd 呼叫三次，再乘 4 個時間點 × 3 個指標。
# twitter 與 superuser 的序列數約為 wiki-vote 的 8 至 12 倍，計算量因此差約 60 至 160 倍，
# 但在足夠的 walltime 內是跑得完的。
#
# 預設 0 = 不設上限，使用全部序列，與 pipeline 內建的評估行為一致。
# 只有在確定跑不完時才設值，且三個資料集要用同一個值，否則數字不可比。
export MMD_MAX_SEQS=${MMD_MAX_SEQS:-0}

cd "${SLURM_SUBMIT_DIR:-$PWD}"

DS=${1:-wiki-vote}
SEED=${2:-123}
KEY="${DS}_${SEED}"
ROOT="../test_and_generated_graphs/${KEY}"

echo "===== 檢查輸入 ====="
MISSING=0
for m in DYMOND AGE DAMNET TagGen; do
    for f in test_graphs.pkl sampled_ts.pkl; do
        if [ -f "${ROOT}/${m}/${f}" ]; then
            printf "  OK   %-8s %-18s %s\n" "$m" "$f" "$(du -h "${ROOT}/${m}/${f}" | cut -f1)"
        else
            printf "  缺   %-8s %s\n" "$m" "$f"
            MISSING=1
        fi
    done
done
[ "$MISSING" -eq 0 ] || { echo "[ERROR] 有缺檔，補齊後再跑"; exit 1; }

echo
echo "===== EVAL ====="
python temporal_baseline_evaluator.py "$KEY" 2>&1 | tee "logs/eval_${KEY}.txt"

echo
echo "結果另存於 logs/eval_${KEY}.txt"
