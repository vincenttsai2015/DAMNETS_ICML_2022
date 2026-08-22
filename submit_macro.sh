#!/bin/bash
# 依 queue 的剩餘名額送出 run_macro_damnets.sh 的 array task。
#
#   bash submit_macro.sh              補到 queue 裡有 20 個（running + pending）
#   bash submit_macro.sh 8            這次最多送 8 個
#   LIMIT=10 bash submit_macro.sh     改成上限 10
#   SEEDS="1 2" bash submit_macro.sh  只送這幾個 seed（多人分工用）
#   ONLY=wiki_vote bash submit_macro.sh   只送名稱含此字串的組合
#   DRY=1 bash submit_macro.sh        只看會送哪些，不實際送出
#
# 已經有結果的組合會自動跳過，跑完一批之後再執行一次就會接著送下一批。
# 組合清單與 run_macro_damnets.sh 完全相同，index 才對得上。

set -eo pipefail
cd "$(dirname "$0")"

LIMIT=${LIMIT:-20}
SEEDS=${SEEDS:-}
RESULT_ROOT=../test_and_generated_graphs

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

IN_QUEUE=$( { squeue -u "$USER" -h 2>/dev/null || true; } | wc -l )
SLOTS=$((LIMIT - IN_QUEUE))
if [ -n "$1" ] && [ "$1" -lt "$SLOTS" ]; then
    SLOTS=$1
fi

echo "queue 現有 ${IN_QUEUE} 個，上限 ${LIMIT}，這次可送 ${SLOTS} 個"
if [ "$SLOTS" -le 0 ]; then
    echo "沒有名額，等前面跑完再執行一次。"
    exit 0
fi

# 掃出還沒有產出的組合
TODO=""
N_TODO=0
N_DONE=0
N_SKIP=0
IDX=0
while read -r group mode model seed; do
    [ -n "$group" ] || continue
    CUR=$IDX
    IDX=$((IDX + 1))
    if [ -n "$ONLY" ] && [ "${group#*$ONLY}" = "$group" ]; then
        N_SKIP=$((N_SKIP + 1))
        continue
    fi
    # SEEDS 限定要跑哪些 seed，多人分工時各自負責不同的 seed
    if [ -n "$SEEDS" ]; then
        case " $SEEDS " in
            *" $seed "*) ;;
            *) N_SKIP=$((N_SKIP + 1)); continue ;;
        esac
    fi
    case "$model" in
        gnn) outdir=DAMNET ;;
        age) outdir=AGE ;;
    esac
    dest="${RESULT_ROOT}/macro_${group}_${mode}_${seed}/${outdir}"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        N_DONE=$((N_DONE + 1))
        continue
    fi
    if [ "$N_TODO" -lt "$SLOTS" ]; then
        TODO="${TODO}${CUR} ${group} ${mode} ${model} ${seed}
"
        N_TODO=$((N_TODO + 1))
    fi
done <<EOF
$COMBOS
EOF

echo "已完成 ${N_DONE}，ONLY/SEEDS 過濾掉 ${N_SKIP}，這次送 ${N_TODO} 個"
if [ "$N_TODO" -eq 0 ]; then
    echo "沒有待跑的組合。"
    exit 0
fi

echo "$TODO" | awk 'NF { printf "  %5d  %-24s %-18s %-6s seed %s\n", $1, $2, $3, $4, $5 }'
LIST=$(echo "$TODO" | awk 'NF { printf "%s,", $1 }' | sed 's/,$//')
echo
echo "index: ${LIST}"

if [ -n "$DRY" ]; then
    echo "（DRY，沒有實際送出）"
    echo "sbatch --array=${LIST} run_macro_damnets.sh"
    exit 0
fi

mkdir -p logs/slurm
sbatch --array="${LIST}" run_macro_damnets.sh
echo
echo "跑完一批之後再執行一次 submit_macro.sh 就會接著送下一批。"
