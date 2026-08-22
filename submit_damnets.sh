#!/bin/bash
# 在登入節點執行（不是 sbatch）。
#
#   bash submit_damnets.sh                                 3 資料集 x 3 seed = 9 個 job
#   bash submit_damnets.sh "wiki-vote" "123 456 789"       指定資料集與 seed
#   bash submit_damnets.sh "wiki-vote twitter" "123"
#
# 一個 job = 一整條 pipeline（前處理 -> DAMNETS -> AGE -> TagGen -> DYMOND -> 評估）。
#
# 為什麼不用 --dependency 把五個步驟拆開送：
#   TWCC 的 20 個名額連 pending 都算，拆開送會讓一條 pipeline 就吃掉 5 個名額。
#
# 前處理只需要做一次。若某個資料集的資料還不存在，該資料集的第一個 seed 會負責產生，
# 其餘 seed 用 --dependency 等它完成，避免同時重複產生同一份資料。

set -eo pipefail
cd ~/DAMNETS_ICML_2022

DATASETS=${1:-"wiki-vote twitter superuser"}
SEEDS=${2:-"123 456 789"}

bins_of() {
    case "$1" in
        wiki-vote) echo 25600  ;;
        twitter)   echo 200000 ;;
        superuser) echo 320000 ;;
        digg)      echo 500000 ;;
        *) echo "" ;;
    esac
}

N_SEEDS=$(echo $SEEDS | wc -w)
N_NEW=0
for DS in $DATASETS; do
    BINS=$(bins_of "$DS")
    [ -n "$BINS" ] || { echo "[ERROR] 未知的資料集: $DS"; exit 1; }
    N_NEW=$(( N_NEW + N_SEEDS ))
    [ -f "data/${DS}/nx_temporal_${DS}_${BINS}_bins_4_timestamps_4_winlen.pkl" ] \
        || N_NEW=$(( N_NEW + 1 ))   # 還要一個獨立的前處理 job
done

N_EXIST=$(squeue -u "$USER" -h | wc -l)
echo "queue 現有 ${N_EXIST} 個 job（pending 也算），本次要送 ${N_NEW} 個，上限 20"
if [ $(( N_EXIST + N_NEW )) -gt 20 ]; then
    echo
    echo "[ERROR] 會超過上限。建議分批，例如一次一個資料集："
    for d in $DATASETS; do echo "    bash submit_damnets.sh \"$d\" \"$SEEDS\""; done
    exit 1
fi
echo

for DS in $DATASETS; do
    BINS=$(bins_of "$DS")
    PKL="data/${DS}/nx_temporal_${DS}_${BINS}_bins_4_timestamps_4_winlen.pkl"

    # 前處理獨立成一個 job，三條 pipeline 只等它，不是等第一個 seed 整條跑完
    DEP=""
    if [ ! -f "$PKL" ]; then
        JP=$(sbatch --parsable --job-name="prep_${DS}" run_prepare_data.sh "$DS")
        printf "  %-24s job %s  <- 產生共用資料\n" "prep_${DS}" "$JP"
        DEP="--dependency=afterok:${JP}"
    fi

    for S in $SEEDS; do
        JID=$(sbatch --parsable $DEP --job-name="dmn_${DS}_${S}" \
                     run_damnets_pipeline.sh "$DS" "$S")
        NOTE=$([ -n "$DEP" ] && echo "  <- 等前處理" || echo "")
        printf "  %-24s job %s%s\n" "${DS}_${S}" "$JID" "$NOTE"
    done
done

echo
echo "追蹤： squeue -u \$USER"
echo "結果： logs/eval_<dataset>_<seed>.txt"
