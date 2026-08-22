#!/bin/bash
#SBATCH --job-name=dmn_pipe
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
# 一個 job 跑完整條 pipeline：前處理（缺才做）-> DAMNETS -> AGE -> TagGen -> DYMOND -> 評估
#
#   sbatch run_damnets_pipeline.sh <dataset> <seed>
#
# 用一個 job 而不是五個，是因為 TWCC 的 20 個名額連 pending 都算，
# 用 --dependency 串起來的 pipeline 會把名額吃光。
#
# 代價是 DAMNETS 與 AGE 改成序列執行，wall time 變長但只佔一個名額。

set -eo pipefail
module load miniconda3/conda24.5.0_py3.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate damnets
export PYTHONNOUSERSITE=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DS=${1:-wiki-vote}
SEED=${2:-123}

case "$DS" in
    wiki-vote) BINS=25600  ;;
    twitter)   BINS=200000 ;;
    superuser) BINS=320000 ;;
    digg)      BINS=500000 ;;
    *) echo "[ERROR] 未知的資料集: $DS"; exit 1 ;;
esac

DATA_PKL="data/${DS}/nx_temporal_${DS}_${BINS}_bins_4_timestamps_4_winlen.pkl"

echo "##################################################"
echo "# pipeline  ${DS}  seed=${SEED}"
echo "# 開始 $(date '+%F %T')"
echo "##################################################"

mkdir -p logs/slurm

# ---------- 1. 前處理 ----------
if [ -f "$DATA_PKL" ]; then
    echo
    echo "########## [1/5] 前處理：已存在，略過 ##########"
    ls -la "$DATA_PKL"
else
    echo
    echo "########## [1/5] 前處理 ##########"
    bash run_prepare_data.sh "$DS"
fi

# ---------- 2. DAMNETS ----------
echo
echo "########## [2/5] DAMNETS ##########"
bash run_damnets.sh "$DS" gnn "$SEED"

# ---------- 3. AGE ----------
echo
echo "########## [3/5] AGE ##########"
bash run_damnets.sh "$DS" age "$SEED"

# ---------- 4. TagGen + DYMOND ----------
echo
echo "########## [4/5] TagGen + DYMOND ##########"
bash run_baselines.sh "$DS" "$SEED"

# ---------- 5. 評估 ----------
echo
echo "########## [5/5] 評估 ##########"
bash run_eval.sh "$DS" "$SEED"

echo
echo "##################################################"
echo "# pipeline 完成  ${DS}  seed=${SEED}"
echo "# 結束 $(date '+%F %T')"
echo "# 結果: logs/eval_${DS}_${SEED}.txt"
echo "##################################################"
echo
echo "--- 磁碟與檔案數 ---"
du -sh experiment_files ../test_and_generated_graphs 2>/dev/null || true
echo "experiment_files 檔案數: $(find experiment_files -type f 2>/dev/null | wc -l)"
echo "結果目錄檔案數:          $(find ../test_and_generated_graphs -type f 2>/dev/null | wc -l)"
exit 0
