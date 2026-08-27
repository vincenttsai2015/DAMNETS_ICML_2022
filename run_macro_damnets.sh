#!/bin/bash
#SBATCH --job-name=dmn_macro
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
# 對注入巨觀動態的資料跑 DAMNETS 與 AGE，接著跑 TagGen 與 DYMOND。
# 後兩者吃的是 DAMNETS 產出的 test_graphs.pkl，所以放在同一個 task 裡依序執行。
#
# 組合清單直接寫在下面的 COMBOS，一行一個：<資料集>_<層> <mode> <model> <seed>。
#
# 一般用 submit_macro.sh 送，它會依 queue 剩餘名額算出這次要送哪些 index。
# 要自己指定範圍的話：
#
#   sbatch --array=0-19 run_macro_damnets.sh          index 0~19
#   sbatch --array=3,7,11 run_macro_damnets.sh        指定幾個
#   bash run_macro_damnets.sh --list                  印出組合對照表
#
# 環境變數：
#   SEEDS="0 1 2"       覆寫 seed 清單，預設 0..9
#   ONLY=wiki_vote      只取名稱含此字串的組合
#   WITH_BASELINES=0    跳過 TagGen 與 DYMOND
#
# 組合順序是 seed 最外層，所以最前面那批剛好是「每個組合各一個 seed」，
# 適合先確認流程；要加 seed 就往後延伸，已跑過的 index 不會變動。

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# 一行一個組合：<資料集>_<層> <mode> <model> <seed>
# 順序是 seed 最外層，所以最前面 64 行是「每個組合各一個 seed」
COMBOS='superuser_a2q raw gnn 0
superuser_a2q raw age 0
superuser_a2q burst gnn 0
superuser_a2q burst age 0
superuser_a2q hysteresis gnn 0
superuser_a2q hysteresis age 0
superuser_a2q burst_hysteresis gnn 0
superuser_a2q burst_hysteresis age 0
superuser_c2a raw gnn 0
superuser_c2a raw age 0
superuser_c2a burst gnn 0
superuser_c2a burst age 0
superuser_c2a hysteresis gnn 0
superuser_c2a hysteresis age 0
superuser_c2a burst_hysteresis gnn 0
superuser_c2a burst_hysteresis age 0
superuser_c2q raw gnn 0
superuser_c2q raw age 0
superuser_c2q burst gnn 0
superuser_c2q burst age 0
superuser_c2q hysteresis gnn 0
superuser_c2q hysteresis age 0
superuser_c2q burst_hysteresis gnn 0
superuser_c2q burst_hysteresis age 0
twitter_MT raw gnn 0
twitter_MT raw age 0
twitter_MT burst gnn 0
twitter_MT burst age 0
twitter_MT hysteresis gnn 0
twitter_MT hysteresis age 0
twitter_MT burst_hysteresis gnn 0
twitter_MT burst_hysteresis age 0
twitter_RT raw gnn 0
twitter_RT raw age 0
twitter_RT burst gnn 0
twitter_RT burst age 0
twitter_RT hysteresis gnn 0
twitter_RT hysteresis age 0
twitter_RT burst_hysteresis gnn 0
twitter_RT burst_hysteresis age 0
wiki_vote_neutral raw gnn 0
wiki_vote_neutral raw age 0
wiki_vote_neutral burst gnn 0
wiki_vote_neutral burst age 0
wiki_vote_neutral hysteresis gnn 0
wiki_vote_neutral hysteresis age 0
wiki_vote_neutral burst_hysteresis gnn 0
wiki_vote_neutral burst_hysteresis age 0
wiki_vote_oppose raw gnn 0
wiki_vote_oppose raw age 0
wiki_vote_oppose burst gnn 0
wiki_vote_oppose burst age 0
wiki_vote_oppose hysteresis gnn 0
wiki_vote_oppose hysteresis age 0
wiki_vote_oppose burst_hysteresis gnn 0
wiki_vote_oppose burst_hysteresis age 0
wiki_vote_support raw gnn 0
wiki_vote_support raw age 0
wiki_vote_support burst gnn 0
wiki_vote_support burst age 0
wiki_vote_support hysteresis gnn 0
wiki_vote_support hysteresis age 0
wiki_vote_support burst_hysteresis gnn 0
wiki_vote_support burst_hysteresis age 0
superuser_a2q raw gnn 1
superuser_a2q raw age 1
superuser_a2q burst gnn 1
superuser_a2q burst age 1
superuser_a2q hysteresis gnn 1
superuser_a2q hysteresis age 1
superuser_a2q burst_hysteresis gnn 1
superuser_a2q burst_hysteresis age 1
superuser_c2a raw gnn 1
superuser_c2a raw age 1
superuser_c2a burst gnn 1
superuser_c2a burst age 1
superuser_c2a hysteresis gnn 1
superuser_c2a hysteresis age 1
superuser_c2a burst_hysteresis gnn 1
superuser_c2a burst_hysteresis age 1
superuser_c2q raw gnn 1
superuser_c2q raw age 1
superuser_c2q burst gnn 1
superuser_c2q burst age 1
superuser_c2q hysteresis gnn 1
superuser_c2q hysteresis age 1
superuser_c2q burst_hysteresis gnn 1
superuser_c2q burst_hysteresis age 1
twitter_MT raw gnn 1
twitter_MT raw age 1
twitter_MT burst gnn 1
twitter_MT burst age 1
twitter_MT hysteresis gnn 1
twitter_MT hysteresis age 1
twitter_MT burst_hysteresis gnn 1
twitter_MT burst_hysteresis age 1
twitter_RT raw gnn 1
twitter_RT raw age 1
twitter_RT burst gnn 1
twitter_RT burst age 1
twitter_RT hysteresis gnn 1
twitter_RT hysteresis age 1
twitter_RT burst_hysteresis gnn 1
twitter_RT burst_hysteresis age 1
wiki_vote_neutral raw gnn 1
wiki_vote_neutral raw age 1
wiki_vote_neutral burst gnn 1
wiki_vote_neutral burst age 1
wiki_vote_neutral hysteresis gnn 1
wiki_vote_neutral hysteresis age 1
wiki_vote_neutral burst_hysteresis gnn 1
wiki_vote_neutral burst_hysteresis age 1
wiki_vote_oppose raw gnn 1
wiki_vote_oppose raw age 1
wiki_vote_oppose burst gnn 1
wiki_vote_oppose burst age 1
wiki_vote_oppose hysteresis gnn 1
wiki_vote_oppose hysteresis age 1
wiki_vote_oppose burst_hysteresis gnn 1
wiki_vote_oppose burst_hysteresis age 1
wiki_vote_support raw gnn 1
wiki_vote_support raw age 1
wiki_vote_support burst gnn 1
wiki_vote_support burst age 1
wiki_vote_support hysteresis gnn 1
wiki_vote_support hysteresis age 1
wiki_vote_support burst_hysteresis gnn 1
wiki_vote_support burst_hysteresis age 1
superuser_a2q raw gnn 2
superuser_a2q raw age 2
superuser_a2q burst gnn 2
superuser_a2q burst age 2
superuser_a2q hysteresis gnn 2
superuser_a2q hysteresis age 2
superuser_a2q burst_hysteresis gnn 2
superuser_a2q burst_hysteresis age 2
superuser_c2a raw gnn 2
superuser_c2a raw age 2
superuser_c2a burst gnn 2
superuser_c2a burst age 2
superuser_c2a hysteresis gnn 2
superuser_c2a hysteresis age 2
superuser_c2a burst_hysteresis gnn 2
superuser_c2a burst_hysteresis age 2
superuser_c2q raw gnn 2
superuser_c2q raw age 2
superuser_c2q burst gnn 2
superuser_c2q burst age 2
superuser_c2q hysteresis gnn 2
superuser_c2q hysteresis age 2
superuser_c2q burst_hysteresis gnn 2
superuser_c2q burst_hysteresis age 2
twitter_MT raw gnn 2
twitter_MT raw age 2
twitter_MT burst gnn 2
twitter_MT burst age 2
twitter_MT hysteresis gnn 2
twitter_MT hysteresis age 2
twitter_MT burst_hysteresis gnn 2
twitter_MT burst_hysteresis age 2
twitter_RT raw gnn 2
twitter_RT raw age 2
twitter_RT burst gnn 2
twitter_RT burst age 2
twitter_RT hysteresis gnn 2
twitter_RT hysteresis age 2
twitter_RT burst_hysteresis gnn 2
twitter_RT burst_hysteresis age 2
wiki_vote_neutral raw gnn 2
wiki_vote_neutral raw age 2
wiki_vote_neutral burst gnn 2
wiki_vote_neutral burst age 2
wiki_vote_neutral hysteresis gnn 2
wiki_vote_neutral hysteresis age 2
wiki_vote_neutral burst_hysteresis gnn 2
wiki_vote_neutral burst_hysteresis age 2
wiki_vote_oppose raw gnn 2
wiki_vote_oppose raw age 2
wiki_vote_oppose burst gnn 2
wiki_vote_oppose burst age 2
wiki_vote_oppose hysteresis gnn 2
wiki_vote_oppose hysteresis age 2
wiki_vote_oppose burst_hysteresis gnn 2
wiki_vote_oppose burst_hysteresis age 2
wiki_vote_support raw gnn 2
wiki_vote_support raw age 2
wiki_vote_support burst gnn 2
wiki_vote_support burst age 2
wiki_vote_support hysteresis gnn 2
wiki_vote_support hysteresis age 2
wiki_vote_support burst_hysteresis gnn 2
wiki_vote_support burst_hysteresis age 2
superuser_a2q raw gnn 3
superuser_a2q raw age 3
superuser_a2q burst gnn 3
superuser_a2q burst age 3
superuser_a2q hysteresis gnn 3
superuser_a2q hysteresis age 3
superuser_a2q burst_hysteresis gnn 3
superuser_a2q burst_hysteresis age 3
superuser_c2a raw gnn 3
superuser_c2a raw age 3
superuser_c2a burst gnn 3
superuser_c2a burst age 3
superuser_c2a hysteresis gnn 3
superuser_c2a hysteresis age 3
superuser_c2a burst_hysteresis gnn 3
superuser_c2a burst_hysteresis age 3
superuser_c2q raw gnn 3
superuser_c2q raw age 3
superuser_c2q burst gnn 3
superuser_c2q burst age 3
superuser_c2q hysteresis gnn 3
superuser_c2q hysteresis age 3
superuser_c2q burst_hysteresis gnn 3
superuser_c2q burst_hysteresis age 3
twitter_MT raw gnn 3
twitter_MT raw age 3
twitter_MT burst gnn 3
twitter_MT burst age 3
twitter_MT hysteresis gnn 3
twitter_MT hysteresis age 3
twitter_MT burst_hysteresis gnn 3
twitter_MT burst_hysteresis age 3
twitter_RT raw gnn 3
twitter_RT raw age 3
twitter_RT burst gnn 3
twitter_RT burst age 3
twitter_RT hysteresis gnn 3
twitter_RT hysteresis age 3
twitter_RT burst_hysteresis gnn 3
twitter_RT burst_hysteresis age 3
wiki_vote_neutral raw gnn 3
wiki_vote_neutral raw age 3
wiki_vote_neutral burst gnn 3
wiki_vote_neutral burst age 3
wiki_vote_neutral hysteresis gnn 3
wiki_vote_neutral hysteresis age 3
wiki_vote_neutral burst_hysteresis gnn 3
wiki_vote_neutral burst_hysteresis age 3
wiki_vote_oppose raw gnn 3
wiki_vote_oppose raw age 3
wiki_vote_oppose burst gnn 3
wiki_vote_oppose burst age 3
wiki_vote_oppose hysteresis gnn 3
wiki_vote_oppose hysteresis age 3
wiki_vote_oppose burst_hysteresis gnn 3
wiki_vote_oppose burst_hysteresis age 3
wiki_vote_support raw gnn 3
wiki_vote_support raw age 3
wiki_vote_support burst gnn 3
wiki_vote_support burst age 3
wiki_vote_support hysteresis gnn 3
wiki_vote_support hysteresis age 3
wiki_vote_support burst_hysteresis gnn 3
wiki_vote_support burst_hysteresis age 3
superuser_a2q raw gnn 4
superuser_a2q raw age 4
superuser_a2q burst gnn 4
superuser_a2q burst age 4
superuser_a2q hysteresis gnn 4
superuser_a2q hysteresis age 4
superuser_a2q burst_hysteresis gnn 4
superuser_a2q burst_hysteresis age 4
superuser_c2a raw gnn 4
superuser_c2a raw age 4
superuser_c2a burst gnn 4
superuser_c2a burst age 4
superuser_c2a hysteresis gnn 4
superuser_c2a hysteresis age 4
superuser_c2a burst_hysteresis gnn 4
superuser_c2a burst_hysteresis age 4
superuser_c2q raw gnn 4
superuser_c2q raw age 4
superuser_c2q burst gnn 4
superuser_c2q burst age 4
superuser_c2q hysteresis gnn 4
superuser_c2q hysteresis age 4
superuser_c2q burst_hysteresis gnn 4
superuser_c2q burst_hysteresis age 4
twitter_MT raw gnn 4
twitter_MT raw age 4
twitter_MT burst gnn 4
twitter_MT burst age 4
twitter_MT hysteresis gnn 4
twitter_MT hysteresis age 4
twitter_MT burst_hysteresis gnn 4
twitter_MT burst_hysteresis age 4
twitter_RT raw gnn 4
twitter_RT raw age 4
twitter_RT burst gnn 4
twitter_RT burst age 4
twitter_RT hysteresis gnn 4
twitter_RT hysteresis age 4
twitter_RT burst_hysteresis gnn 4
twitter_RT burst_hysteresis age 4
wiki_vote_neutral raw gnn 4
wiki_vote_neutral raw age 4
wiki_vote_neutral burst gnn 4
wiki_vote_neutral burst age 4
wiki_vote_neutral hysteresis gnn 4
wiki_vote_neutral hysteresis age 4
wiki_vote_neutral burst_hysteresis gnn 4
wiki_vote_neutral burst_hysteresis age 4
wiki_vote_oppose raw gnn 4
wiki_vote_oppose raw age 4
wiki_vote_oppose burst gnn 4
wiki_vote_oppose burst age 4
wiki_vote_oppose hysteresis gnn 4
wiki_vote_oppose hysteresis age 4
wiki_vote_oppose burst_hysteresis gnn 4
wiki_vote_oppose burst_hysteresis age 4
wiki_vote_support raw gnn 4
wiki_vote_support raw age 4
wiki_vote_support burst gnn 4
wiki_vote_support burst age 4
wiki_vote_support hysteresis gnn 4
wiki_vote_support hysteresis age 4
wiki_vote_support burst_hysteresis gnn 4
wiki_vote_support burst_hysteresis age 4'

N_COMBOS=320

if [ "$1" = "--list" ]; then
    echo "共 ${N_COMBOS} 個組合"
    printf "%5s  %-24s %-18s %-6s %s
" idx group mode model seed
    echo "$COMBOS" | awk '{ printf "%5d  %-24s %-18s %-6s %s\n", NR-1, $1, $2, $3, $4 }'
    exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:?這支要用 sbatch 送，或加 --list 看組合表}
[ "$IDX" -lt "$N_COMBOS" ] || { echo "[ERROR] index $IDX 超出範圍（共 $N_COMBOS 個）"; exit 1; }

LINE=$(echo "$COMBOS" | sed -n "$((IDX + 1))p")
GROUP=$(echo "$LINE" | awk '{print $1}')
MODE=$(echo "$LINE" | awk '{print $2}')
MODEL=$(echo "$LINE" | awk '{print $3}')
SEED=$(echo "$LINE" | awk '{print $4}')
DS="macro_${GROUP}_${MODE}"

echo "=================================================="
echo " array index ${IDX} / ${N_COMBOS}"
echo " dataset : ${DS}"
echo " model   : ${MODEL}"
echo " seed    : ${SEED}"
echo "=================================================="

bash run_damnets.sh "$DS" "$MODEL" "$SEED"

# TagGen 與 DYMOND 不需要訓練，直接吃剛產出的 test_graphs.pkl。
# 只在 gnn 那一輪跑，避免 age 那輪重複一次。
if [ "${WITH_BASELINES:-1}" = "1" ] && [ "$MODEL" = "gnn" ]; then
    echo
    echo "===== DYMOND ====="
    # DYMOND 要吃觀測窗而不是 target 段（它是擬合統計再重生，給後半等於看到
    # 答案）。觀測窗那份的檔名是 nx_macro_<資料集>_<mode>_L<層編號>_<層名>，
    # 層編號在組合表裡沒有，用層名反查。
    LAYER=${GROUP#*_}
    DSNAME=${GROUP%%_*}
    case "$GROUP" in
        wiki_vote_*) DSNAME=wiki_vote; LAYER=${GROUP#wiki_vote_} ;;
    esac
    OBS_DIR=${OBS_DIR:-data/macro_obs}
    OBS=$(ls "${OBS_DIR}"/nx_macro_"${DSNAME}"_"${MODE}"_L*_"${LAYER}".pkl 2>/dev/null | head -1)
    if [ -n "$OBS" ]; then
        export DYMOND_OBS="$OBS"
        echo "觀測窗: $DYMOND_OBS"
    else
        echo "[WARN] ${OBS_DIR} 底下找不到 ${DSNAME}_${MODE}_*_${LAYER}，DYMOND 會吃到 target 段"
    fi
    bash run_baselines.sh "$DS" "$SEED"
fi
