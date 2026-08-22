#!/bin/bash
#SBATCH --job-name=dmn_train
#SBATCH --account=ACD109125
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#
# AGE 或 DAMNETS 的 training + sampling，跑完把結果放進評估器讀的位置。
#
#   sbatch run_damnets.sh wiki-vote gnn 123     # DAMNETS
#   sbatch run_damnets.sh wiki-vote age 123     # AGE
#
# model 只能是 gnn 或 age。gnn -> DAMNET 目錄，age -> AGE 目錄。

set -eo pipefail
module load miniconda3/conda24.5.0_py3.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate damnets
export PYTHONNOUSERSITE=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DS=${1:-wiki-vote}
MODEL=${2:-gnn}
SEED=${3:-123}

case "$MODEL" in
    gnn) OUTDIR=DAMNET; NAME=DAMNETS ;;
    age) OUTDIR=AGE;    NAME=AGE ;;
    *) echo "[ERROR] model 只能是 gnn 或 age，收到: $MODEL"; exit 1 ;;
esac

RUNNAME="${NAME}_${DS}_${SEED}"
BASE_CFG="experiment_configs/${DS}_${MODEL}.yaml"
CFG_NAME="${DS}_${MODEL}_${SEED}.yaml"
CFG="experiment_configs/${CFG_NAME}"

# 評估器的路徑是寫死的 ../test_and_generated_graphs/<argv1>/<MODEL>/，
# 而 <MODEL> 只能是 DYMOND / AGE / DAMNET / TagGen 四個固定名稱，
# 所以 seed 只能放進 dataset 那一層。評估時 argv1 要傳 <ds>_<seed>。
DEST="../test_and_generated_graphs/${DS}_${SEED}/${OUTDIR}"

echo "=================================================="
echo " ${RUNNAME}"
echo " config : ${CFG}"
echo " output : ${DEST}"
echo "=================================================="

[ -f "$BASE_CFG" ] || { echo "[ERROR] 找不到 $BASE_CFG"; exit 1; }
mkdir -p logs/slurm experiment_files "$DEST"

# ---------- 產生本次的 config ----------
# repo 內的 config 全部是 CRLF。不先轉成 LF 的話，底下 awk 取出來的路徑
# 會帶著行尾的 \r，[ -f ] 必定判定為不存在。
cp "$BASE_CFG" "$CFG"
sed -i 's/\r$//' "$CFG"
sed -i "s/^seed: .*/seed: ${SEED}/" "$CFG"
echo "--- config 生效值 ---"
grep -nE '^seed:|^  (max_n|N|T|epochs|batch_size):|^  path:' "$CFG" || true

DATA_PATH=$(grep -E '^  path:' "$CFG" | tr -d '\r' | awk '{print $2}')
[ -f "$DATA_PATH" ] || { echo "[ERROR] 資料檔不存在: $DATA_PATH（先跑 run_prepare_data.sh）"; exit 1; }

# ---------- 訓練 ----------
# get_config() 的 exp_name 結尾是 python 行程的 PID（config.run_id = os.getpid()），
# 所以用 $! 就能精準定位本次產生的目錄，多個 job 同時跑也不會抓錯。
echo
echo "===== TRAIN ====="
python run_exp.py -c "$CFG_NAME" &
TRAIN_PID=$!
wait "$TRAIN_PID"

TRAIN_DIR=$(ls -d experiment_files/*_"${TRAIN_PID}" 2>/dev/null | head -1)
[ -n "$TRAIN_DIR" ] && [ -d "$TRAIN_DIR" ] \
    || { echo "[ERROR] 找不到 PID ${TRAIN_PID} 對應的訓練目錄"; exit 1; }
echo "===== TRAIN_DIR = ${TRAIN_DIR} ====="

grep -nE 'best_val_epoch|graph_dir' "$TRAIN_DIR/config.yaml" \
    || { echo "[ERROR] config 沒有 best_val_epoch，代表 validation 一次都沒跑到。"
         echo "        validation 條件是 epoch % val_epochs == 0 且 epoch > 0，"
         echo "        所以 epochs 必須大於 val_epochs。"; exit 1; }

# ---------- 取樣 ----------
echo
echo "===== SAMPLE ====="
python run_exp.py -t -c "$TRAIN_DIR/config.yaml" &
TEST_PID=$!
wait "$TEST_PID"

TEST_DIR=$(ls -d experiment_files/*_"${TEST_PID}" 2>/dev/null | head -1)
[ -n "$TEST_DIR" ] && [ -d "$TEST_DIR" ] \
    || { echo "[ERROR] 找不到 PID ${TEST_PID} 對應的取樣目錄"; exit 1; }
echo "===== TEST_DIR = ${TEST_DIR} ====="

# ---------- 放到評估器讀的位置 ----------
cp "$TRAIN_DIR/test_graphs.pkl" "$DEST/test_graphs.pkl"
cp "$TEST_DIR/sampled_ts.pkl"   "$DEST/sampled_ts.pkl"

# ---------- 清掉逐對樣本的快取 ----------
# GNNTSampler / TFTSampler 會把每一對 (G_t, G_t+1) 各寫成一個 pickle，
# 數量是 N x (T-1)。twitter 的 N=50000 就是十幾萬個小檔，superuser 更多。
# 這些跑完就沒用了，留著會吃掉大量 inode。設 KEEP_CACHE=1 可保留。
if [ "${KEEP_CACHE:-0}" != "1" ]; then
    for d in "$TRAIN_DIR/data_cache" "$TEST_DIR/data_cache"; do
        if [ -d "$d" ]; then
            n=$(find "$d" -type f | wc -l)
            rm -rf "$d"
            echo "已清除快取 ${d}（${n} 個檔）"
        fi
    done
fi

echo
echo "===== DONE  ${RUNNAME} ====="
ls -la "$DEST"
