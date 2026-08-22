#!/bin/bash
# 掃出哪些組合缺哪個模型的產出，並印出補跑指令。
#
#   bash check_baselines.sh              列出缺漏
#   bash check_baselines.sh --submit     直接把缺的補送出去（受 LIMIT 名額限制）
#   LIMIT=10 bash check_baselines.sh --submit
#   ONLY=wiki_vote bash check_baselines.sh
#   SEED_TAG=1 bash check_baselines.sh    只看 seed 1（多人分工用）
#
# 判斷方式與 submit_macro.sh 一致：目錄存在且非空才算完成。
# TagGen 與 DYMOND 是掛在 DAMNETS 之後跑的，DAMNETS 成功但這兩個失敗時，
# submit_macro.sh 會把該組合當成已完成而跳過，所以要用這支單獨補。

set -eo pipefail
cd "$(dirname "$0")"

RESULT_ROOT=../test_and_generated_graphs
# 要納入比較的模型。TagGen 對每一條序列各自訓練一個 transformer，
# superuser 一組要 24 小時以上，佔了整批九成以上的時間，因此移出預設。
# 要加回來：BASELINE_MODELS="DAMNET AGE TagGen DYMOND"
MODELS=${BASELINE_MODELS:-"DAMNET AGE DYMOND"}
LIMIT=${LIMIT:-20}
SEED_TAG=${SEED_TAG:-}
SUBMIT=0
[ "$1" = "--submit" ] && SUBMIT=1

# queue 裡還在跑的組合先跳過。TagGen 與 DYMOND 掛在 DAMNETS 之後，
# job 還沒走到那一段時目錄本來就是空的，不算失敗。
RUNNING=""
if command -v squeue >/dev/null 2>&1; then
    N_RUNNING=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
    if [ "$N_RUNNING" -gt 0 ]; then
        echo "queue 裡有 ${N_RUNNING} 個 job。"
        # scontrol 的 Command= 會連參數一起印，補跑 job 直接讀得出組合，
        # 不論它是用哪個版本送出去的。
        for j in $(squeue -u "$USER" -h -o "%i" 2>/dev/null); do
            cmd=$(scontrol show job "$j" 2>/dev/null | grep -o 'Command=.*' | head -1)
            cmd=${cmd#Command=}
            set -- $cmd
            case "$1" in
                *run_baselines.sh)
                    [ -n "$2" ] && RUNNING="${RUNNING} ${2}_${3}" ;;
            esac
        done

        # array job 沒有參數，index 對得回組合，從 run_macro_damnets.sh 的清單反查
        for jid in $(squeue -u "$USER" -h -o "%K" 2>/dev/null | grep -E '^[0-9]+$'); do
            line=$(sed -n "$((jid + 1))p" <(grep -A 100000 "^COMBOS='" run_macro_damnets.sh | tail -n +2))
            g=$(echo "$line" | awk '{print $1}')
            m=$(echo "$line" | awk '{print $2}')
            s=$(echo "$line" | awk '{print $4}')
            [ -n "$g" ] && RUNNING="${RUNNING} macro_${g}_${m}_${s}"
        done
    fi
fi

N_OK=0
N_MISS=0
N_RUN=0
N_DAMNET=0
N_AGE=0
N_TAGGEN=0
N_DYMOND=0
MISSING=""

# SEED_TAG 限定只看某個 seed，留空看全部。多人分工時各自負責不同的 seed
_pat="macro_*"
[ -n "$SEED_TAG" ] && _pat="macro_*_${SEED_TAG}"
DIRS=$(find "$RESULT_ROOT" -maxdepth 1 -type d -name "$_pat" 2>/dev/null | sort)
if [ -z "$DIRS" ]; then
    echo "[ERROR] ${RESULT_ROOT} 底下找不到任何 macro_* 目錄"
    echo "        目前位置：$(pwd)"
    echo "        該目錄的內容："
    ls -d "$RESULT_ROOT"/* 2>/dev/null | head -5 || echo "        （不存在或是空的）"
    exit 1
fi

for d in $DIRS; do
    [ -d "$d" ] || continue
    key=${d##*/}
    if [ -n "$ONLY" ] && [ "${key#*$ONLY}" = "$key" ]; then continue; fi
    case " $RUNNING " in
        *" $key "*) N_RUN=$((N_RUN + 1)); continue ;;
    esac

    # <組合>_<seed>，seed 是最後一段
    seed=${key##*_}
    ds=${key%_*}

    miss=""
    for m in $MODELS; do
        if [ -d "$d/$m" ] && [ -n "$(ls -A "$d/$m" 2>/dev/null)" ]; then
            # 四個模型都把生成結果寫成 sampled_ts.pkl。目錄非空不代表跑完，
            # run_baselines.sh 一開始就把 test_graphs.pkl 複製進去了。
            out="$d/$m/sampled_ts.pkl"
            if [ -f "$out" ]; then
                case "$m" in
                    DAMNET) N_DAMNET=$((N_DAMNET + 1)) ;;
                    AGE)    N_AGE=$((N_AGE + 1)) ;;
                    TagGen) N_TAGGEN=$((N_TAGGEN + 1)) ;;
                    DYMOND) N_DYMOND=$((N_DYMOND + 1)) ;;
                esac
            else
                miss="$miss $m"
            fi
        else
            miss="$miss $m"
        fi
    done

    if [ -z "$miss" ]; then
        N_OK=$((N_OK + 1))
    else
        N_MISS=$((N_MISS + 1))
        printf '%-52s 缺:%s\n' "$key" "$miss"
        # 只有 TagGen / DYMOND 缺的話用 run_baselines.sh 補
        case "$miss" in
            *DAMNET*|*AGE*)
                MISSING="${MISSING}# ${key} 缺 DAMNETS/AGE，要用 submit_macro.sh 重跑
" ;;
            *)
                # miss 只會是 MODELS 的子集，TagGen 不在預設清單裡
                which="dymond"
                case "$miss" in *TagGen*) which="both" ;; esac
                MISSING="${MISSING}sbatch --job-name=bf_${key} run_baselines.sh ${ds} ${seed} ${which}
" ;;
        esac
    fi
done

N_SCANNED=$((N_OK + N_MISS))
echo
echo "掃描 ${N_SCANNED} 組（另有 ${N_RUN} 組執行中不列入）"
echo "  四個模型都齊 ${N_OK} 組，有缺 ${N_MISS} 組"
echo
echo "  各模型的產出數（分母 ${N_SCANNED}）："
printf '    %-8s %d
' DAMNETS "$N_DAMNET" AGE "$N_AGE" TagGen "$N_TAGGEN" DYMOND "$N_DYMOND"
[ "$N_MISS" -eq 0 ] && exit 0

echo
echo "--- 補跑指令 ---"
printf '%s' "$MISSING"

if [ "$SUBMIT" = "1" ]; then
    # 與 submit_macro.sh 同一個上限。超過的話 SLURM 會直接拒絕而不是排隊，
    # 送出失敗的那些不會留下任何痕跡，所以這裡自己數。
    IN_QUEUE=$( { squeue -u "$USER" -h 2>/dev/null || true; } | wc -l )
    SLOTS=$((LIMIT - IN_QUEUE))
    N_CMD=$(printf '%s' "$MISSING" | grep -vc '^#' || true)

    echo
    echo "queue 現有 ${IN_QUEUE} 個，上限 ${LIMIT}，這次可送 ${SLOTS} 個（待補 ${N_CMD} 個）"
    if [ "$SLOTS" -le 0 ]; then
        echo "沒有名額，等 queue 空出來再執行一次"
        exit 0
    fi

    echo
    echo "--- 送出 ---"
    n=0
    printf '%s' "$MISSING" | grep -v '^#' | while read -r cmd; do
        [ -n "$cmd" ] || continue
        n=$((n + 1))
        [ "$n" -gt "$SLOTS" ] && break
        echo "$cmd"
        eval "$cmd"
    done

    if [ "$N_CMD" -gt "$SLOTS" ]; then
        echo
        echo "還有 $((N_CMD - SLOTS)) 個沒送，等 queue 空出來再執行一次"
    fi
fi
