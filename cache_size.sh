# 一組執行需要的磁碟空間，以及可用空間。
# submit_macro.sh 用它決定併發，run_macro_damnets.sh 用它決定要不要等。
#
#   . ./cache_size.sh
#   cache_gb superuser_a2q   ->  18
#   free_gb                  ->  experiment_files 所在檔案系統的剩餘 GB

# 一次執行寫進 experiment_files/<exp>/ 的峰值。AGE 與兩支測試 sampler 改存
# 邊列之後只剩幾百 MB，現在的大宗是 DAMNETS 訓練的 data_cache：
# int(序列數 x 0.8) x 每條 16 對 x 每對約 0.2 MB。取實測往上抓的上限——
# 低估會把磁碟寫爆，而寫爆的 job 不會執行結尾的清理，殘留讓下一次更容易爆。
# 認不得的名稱一律按最大值抓。
cache_gb() {
    case "$1" in
        superuser_*) echo 18 ;;
        twitter_*)   echo 8 ;;
        wiki_vote_*) echo 4 ;;
        *)           echo 18 ;;
    esac
}

# 量的是 experiment_files 實際落在哪個檔案系統。它可能是指向 /work 的符號
# 連結，那時候家目錄的剩餘量跟這件事無關，量錯了會把併發算得太保守。
_exp_path() {
    _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiment_files"
    [ -e "$_d" ] || _d="$(dirname "$_d")"
    echo "$_d"
}

free_gb() {
    df -Pk "${1:-$(_exp_path)}" | awk 'NR==2 { printf "%d", $4 / 1048576 }'
}
