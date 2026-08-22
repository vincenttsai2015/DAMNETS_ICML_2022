#!/bin/bash
# 在 TWCC 執行。把 DAMNETS 這組四個模型的結果打包成一個檔。
#
#   bash pack_results.sh              評估結果 + log + 各 run 的 config
#   bash pack_results.sh --with-pkl   另外含生成的圖序列 pkl（會很大）
#
# 輸出： ~/damnets_results.tar.gz

cd ~/DAMNETS_ICML_2022 || { echo "[ERROR] 找不到 ~/DAMNETS_ICML_2022"; exit 1; }
shopt -s nullglob
OUT=~/damnets_results.tar.gz

EVALS=(logs/eval_*.txt)
echo "===== 評估結果（${#EVALS[@]} 份）====="
for f in "${EVALS[@]}"; do
    echo "--- $f ---"
    grep -E "Average|KS in data" "$f" || tail -20 "$f"
    echo
done
[ ${#EVALS[@]} -eq 0 ] && echo "  還沒有任何評估結果，先看 logs/slurm/*.err"

echo "===== 指標彙整 ====="
python collect_metrics.py --csv > damnets_metrics.csv 2>/dev/null || true
if [ -s damnets_metrics.csv ]; then
    column -s, -t damnets_metrics.csv 2>/dev/null || cat damnets_metrics.csv
else
    echo "  尚無可解析的評估結果"
fi
echo

ITEMS=()
[ -s damnets_metrics.csv ] && ITEMS+=(damnets_metrics.csv)
for f in "${EVALS[@]}"; do ITEMS+=("$f"); done
[ -d logs/slurm ] && ITEMS+=(logs/slurm)

# 每個 run 的訓練統計與 config，不含 data_cache（很大且可重建）
for d in experiment_files/*/; do
    for f in config.yaml train_stats.pkl; do
        [ -f "${d}${f}" ] && ITEMS+=("${d}${f}")
    done
done

if [ "$1" = "--with-pkl" ] && [ -d ../test_and_generated_graphs ]; then
    ITEMS+=(../test_and_generated_graphs)
fi

if [ ${#ITEMS[@]} -eq 0 ]; then
    echo "[ERROR] 沒有任何可打包的內容"
    exit 1
fi

echo "===== 打包 ====="
echo "共 ${#ITEMS[@]} 個項目"
rm -f "$OUT"
tar czf "$OUT" "${ITEMS[@]}" || { echo "[ERROR] 打包失敗"; exit 1; }

echo
ls -lh "$OUT"
echo
echo "本機（WSL）抓回："
echo "  scp roy12358@ln01.twcc.ai:~/damnets_results.tar.gz /mnt/c/Users/User/Desktop/intern/refs/results/"
