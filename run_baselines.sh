#!/bin/bash
#SBATCH --job-name=dmn_base
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
# TagGen 與 DYMOND。兩者都直接吃 test_graphs.pkl，不需要訓練 checkpoint，
# 但需要先跑過 run_damnets.sh 產生 test_graphs.pkl。
#
#   sbatch run_baselines.sh wiki-vote 123
#
# 只跑其中一個：
#   sbatch run_baselines.sh wiki-vote 123 taggen
#   sbatch run_baselines.sh wiki-vote 123 dymond
#
# TagGen 與 DYMOND 的原始碼沒有把 seed 接出來，重跑只會拿到不同的隨機抽樣。

set -eo pipefail
module load miniconda3/conda24.5.0_py3.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate damnets
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# TagGen 對每一條序列各做一次完整訓練，序列數大時單執行緒跑不完。
#   TAGGEN_MAX_SEQS  取幾條序列，0 = 全部（與其他三個模型一致）
#   TAGGEN_WORKERS   平行度，1 = 原本的序列執行
export TAGGEN_MAX_SEQS=${TAGGEN_MAX_SEQS:-0}
export TAGGEN_WORKERS=${TAGGEN_WORKERS:-4}

cd "${SLURM_SUBMIT_DIR:-$PWD}"

DS=${1:-wiki-vote}
SEED=${2:-123}
# TagGen 移出預設，單組要 24 小時以上。要跑的話明確指定 taggen 或 both。
WHICH=${3:-dymond}
KEY="${DS}_${SEED}"
ROOT="../test_and_generated_graphs/${KEY}"

SRC=""
for d in DAMNET AGE; do
    if [ -f "${ROOT}/${d}/test_graphs.pkl" ]; then SRC="${ROOT}/${d}/test_graphs.pkl"; break; fi
done
[ -n "$SRC" ] || { echo "[ERROR] 找不到 test_graphs.pkl，先跑 run_damnets.sh"; exit 1; }
echo "test_graphs.pkl 來源: ${SRC}"

if [ "$WHICH" = "both" ] || [ "$WHICH" = "taggen" ]; then
    mkdir -p "${ROOT}/TagGen"
    cp "$SRC" "${ROOT}/TagGen/test_graphs.pkl"
    echo
    echo "===== TagGen ====="
    echo "注意：run_tag_gen.py 只取最後 5 條序列（原始碼寫死 [-5:]）"
    python run_tag_gen.py "$KEY"
    ls -la "${ROOT}/TagGen"
fi

if [ "$WHICH" = "both" ] || [ "$WHICH" = "dymond" ]; then
    mkdir -p "${ROOT}/DYMOND"
    # DAMNETS 與 DYMOND 要的區間不同。前者是自迴歸的，序列第一張當種子往後生；
    # 後者是對整條擬合統計再重生一條同長度的，給它目標窗等於讓它看過答案。
    # 所以 DYMOND 吃觀測窗（DYMOND_OBS 指向 --t1 16 產出的那份），
    # 生出來的 16 張才對應要預測的目標窗。
    # 兩份的序列順序一致，取一樣多的尾巴就對得上同一批測試序列。
    if [ -n "${DYMOND_OBS:-}" ] && [ -f "${DYMOND_OBS}" ]; then
        echo "DYMOND 觀測窗來源: ${DYMOND_OBS}"
        python - "$SRC" "$DYMOND_OBS" "${ROOT}/DYMOND/test_graphs.pkl" <<'PY'
import pickle, sys
ref, obs, out = sys.argv[1:4]
n = len(pickle.load(open(ref, 'rb')))
full = pickle.load(open(obs, 'rb'))
sel = full[-n:]
pickle.dump(sel, open(out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
print(f'取觀測窗最後 {n} 條（每條 {len(sel[0])} 張）')
PY
    else
        echo "[WARN] 沒設 DYMOND_OBS，沿用 ${SRC}——那是目標窗，DYMOND 會看到答案"
        cp "$SRC" "${ROOT}/DYMOND/test_graphs.pkl"
    fi
    echo
    echo "===== DYMOND ====="
    echo "CWD=$(pwd)  python=$(command -v python)"
    echo "解析到的檔案: $(ls -l run_dymond.py 2>&1)"
    echo "注意：DYMOND 會對每一條序列開一個子目錄並用 multiprocessing Pool 平行處理"
    python run_dymond.py "$KEY"
    ls -la "${ROOT}/DYMOND"
fi

echo
echo "===== DONE ====="
