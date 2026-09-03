"""掃出讀不起來的產出檔，選擇性刪掉那個模型的目錄。

磁碟寫爆或 job 被砍時會留下截斷的 pickle。完成與否只看目錄裡有沒有東西
（submit_macro.sh 的判準），所以截斷檔會被當成已完成而永遠跳過，
一路撐到最後算指標才炸。

    python check_broken.py                掃描並列出
    python check_broken.py --delete       刪掉壞檔所在的模型目錄，之後會重跑
    python check_broken.py --root DIR     指定產出目錄
    python check_broken.py --quiet        只印壞的

產出裡是 networkx 的圖，反序列化需要 networkx。用錯 python（例如沒 activate
環境的 base）會讓每個檔都拋 ModuleNotFoundError，那是環境問題不是檔案壞掉——
腳本會直接中止而不是把全部刪光。
"""
import argparse
import glob
import os
import pickle
import shutil
import sys

try:
    import networkx  # noqa: F401  反序列化產出時需要
except ImportError:
    sys.exit("[ERROR] 這個 python 沒有 networkx，讀不了產出檔。\n"
             "        先 conda activate damnets，或直接用\n"
             "        ~/miniconda3/envs/damnets/bin/python check_broken.py")

# 這幾種例外代表跑的環境不對，不是檔案有問題。碰到就中止。
ENV_ERRORS = (ImportError, ModuleNotFoundError, AttributeError)


def inspect(path):
    """回傳 (狀態, 說明)。狀態是 'ok' / 'bad' / 'env'。"""
    try:
        obj = pickle.load(open(path, "rb"))
    except ENV_ERRORS as e:
        return "env", f"{type(e).__name__}: {e}"
    except Exception as e:
        return "bad", f"{type(e).__name__}: {e}"

    if not isinstance(obj, list) or not obj:
        return "bad", f"頂層是 {type(obj).__name__}，不是非空 list"

    lens = set()
    for seq in obj:
        if not isinstance(seq, list) or not seq:
            return "bad", "有元素不是非空的序列"
        lens.add(len(seq))
        for g in seq:
            if not hasattr(g, "number_of_edges"):
                return "bad", f"序列裡有 {type(g).__name__}，不是圖"

    if len(lens) != 1:
        return "bad", f"每條張數不一致：{sorted(lens)}"

    return "ok", f"{len(obj)} 條 x {lens.pop()} 張，節點 {obj[0][0].number_of_nodes()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../test_and_generated_graphs")
    ap.add_argument("--delete", action="store_true",
                    help="刪掉壞檔所在的模型目錄")
    ap.add_argument("--quiet", action="store_true", help="只印壞的")
    ap.add_argument("--force", action="store_true",
                    help="壞檔比例過高時仍然刪除")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "*", "*", "*.pkl")))
    if not paths:
        sys.exit(f"{args.root} 底下沒有 pkl")

    bad_dirs, n_ok, n_bad = [], 0, 0
    for p in paths:
        state, why = inspect(p)
        rel = os.path.relpath(p, args.root)
        if state == "env":
            sys.exit(f"[ERROR] {rel} 讀取時出現環境問題：{why}\n"
                     "        這代表 python 環境不對，不是檔案壞掉。已中止，"
                     "沒有刪除任何東西。")
        if state == "ok":
            n_ok += 1
            if not args.quiet:
                print(f"OK   {rel}  {why}")
        else:
            n_bad += 1
            print(f"壞   {rel}  {why}")
            bad_dirs.append(os.path.dirname(p))

    bad_dirs = sorted(set(bad_dirs))
    print(f"\n可用 {n_ok}，壞 {n_bad}，牽涉 {len(bad_dirs)} 個模型目錄")

    if not bad_dirs:
        return

    # 全部或幾乎全部都壞，通常表示判斷方式本身有問題，而不是真的全毀。
    if n_ok == 0 and not args.force:
        sys.exit("\n[ERROR] 沒有任何一個檔案可用。這比較像是掃描方式或環境有問題，"
                 "\n        不像是產出真的全毀。先確認幾個檔案再說，"
                 "\n        確定要刪的話加 --force。")

    if not args.delete:
        print("加 --delete 會刪掉這些目錄：")
        for d in bad_dirs:
            print(f"  {d}")
        return

    for d in bad_dirs:
        shutil.rmtree(d, ignore_errors=True)
        print(f"已刪 {d}")


if __name__ == "__main__":
    main()
