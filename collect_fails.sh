#!/bin/bash
# 收集所有非正常結束的 job 與它們的 log 尾段，打包成 fails.tar.gz。
#
#   bash collect_fails.sh                 掃 2026-08-20 之後
#   SINCE=2026-08-18 bash collect_fails.sh
#
# log 完整檔可能很大，只取尾段與錯誤行。

set -eo pipefail
cd "$(dirname "$0")"

SINCE=${SINCE:-2026-08-20}
LOGDIR=${LOGDIR:-logs/slurm}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

# -X 只列主 job，不含 .batch / .extern 那些重複列
sacct -u "$USER" --starttime="$SINCE" -X -n \
      --format=JobID%22,JobName%16,State%20,ExitCode%10,Elapsed%12 \
    > "$OUT/sacct_all.txt"

grep -Ev 'COMPLETED|RUNNING|PENDING' "$OUT/sacct_all.txt" > "$OUT/sacct_bad.txt" || true

N=$(wc -l < "$OUT/sacct_bad.txt")
echo "非正常結束 ${N} 個："
cat "$OUT/sacct_bad.txt"

if [ "$N" -eq 0 ]; then
    echo "沒有要收的"
    exit 0
fi

mkdir -p "$OUT/logs"
MISS=0
while read -r jid rest; do
    [ -n "$jid" ] || continue
    found=0
    for f in "$LOGDIR"/*"${jid}"*; do
        [ -f "$f" ] || continue
        found=1
        {
            echo "===================== ${f} ====================="
            echo "--- 尾段 80 行 ---"
            tail -80 "$f"
        } >> "$OUT/logs/${jid}.txt"
    done
    [ "$found" -eq 0 ] && { echo "找不到 log: $jid"; MISS=$((MISS + 1)); }
done < "$OUT/sacct_bad.txt"

# 目前各組合的產出狀態一起帶上，對照用
bash check_baselines.sh > "$OUT/check.txt" 2>&1 || true

tar czf ~/fails.tar.gz -C "$OUT" sacct_all.txt sacct_bad.txt check.txt logs
echo
echo "已寫出 ~/fails.tar.gz（$(du -h ~/fails.tar.gz | cut -f1)），缺 log ${MISS} 個"
