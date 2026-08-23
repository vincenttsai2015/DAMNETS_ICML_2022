"""比對三個模型生成圖的節點數與空圖比例。

    python check_dymond.py            掃 seed 0 的全部組合
    python check_dymond.py 1          掃 seed 1
    python check_dymond.py 0 wiki     只看名稱含 wiki 的
    CHECK_SAMPLE=0 python check_dymond.py    每個檔讀完整（很慢）

預設每個檔只取前 20 條序列——superuser 一個 sampled_ts.pkl 是 180 MB，
32 組乘上 3 個模型全讀要很久，而節點數一不一致看前幾條就夠了。

DYMOND 原本用 nx.from_edgelist 收集結果，那個只建立有邊的節點——孤立節點
會消失、整張沒邊就成了零節點的圖。DAMNETS 與 AGE 是從固定大小的鄰接矩陣
還原、節點集完整，兩邊對不起來會讓 degree 分佈的直方圖退化，MMD 算出負值。

「無邊」那一欄數的是沒有任何邊的圖，那是正常的（節點集仍然完整，
degree 直方圖是全零），不是問題。要看的是節點數三個模型一不一致。
"""
import glob
import os
import pickle
import sys

MODELS = ["DAMNET", "AGE", "DYMOND"]
ROOT = "../test_and_generated_graphs"
SAMPLE = int(os.environ.get("CHECK_SAMPLE", "20"))


def stats(path):
    with open(path, "rb") as f:
        seqs = pickle.load(f)
    total = len(seqs)
    if SAMPLE:
        seqs = seqs[:SAMPLE]
    n = [g.number_of_nodes() for s in seqs for g in s]
    e = [g.number_of_edges() for s in seqs for g in s]
    empty = sum(1 for x in e if x == 0)
    return total, min(n), max(n), empty, len(e)


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else "0"
    only = sys.argv[2] if len(sys.argv) > 2 else ""

    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, f"macro_*_{seed}"))
                  if os.path.isdir(d) and (not only or only in d))
    if not dirs:
        raise SystemExit(f"找不到 {ROOT}/macro_*_{seed}")

    print(f"{'組合':46s} {'模型':8s} {'序列':>5s} {'節點':>12s} {'無邊':>14s}")
    bad = []
    for d in dirs:
        key = os.path.basename(d)
        ns = {}
        for m in MODELS:
            p = os.path.join(d, m, "sampled_ts.pkl")
            if not os.path.exists(p):
                continue
            try:
                n_seq, lo, hi, empty, total = stats(p)
            except Exception as ex:
                print(f"{key:46s} {m:8s} 讀取失敗 {type(ex).__name__}")
                continue
            ns[m] = (lo, hi)
            rng = f"{lo}~{hi}" if lo != hi else str(lo)
            print(f"{key if m == MODELS[0] else '':46s} {m:8s} {n_seq:>5d} "
                  f"{rng:>12s} {empty:>7d}/{total:<6d}", flush=True)

        # 三個模型的節點數範圍要一致
        if len(ns) == len(MODELS) and len(set(ns.values())) > 1:
            bad.append((key, ns))
        print()

    print("=" * 70)
    if bad:
        print(f"{len(bad)} 組的節點數對不起來：")
        for key, ns in bad:
            print(f"  {key}")
            for m, (lo, hi) in ns.items():
                print(f"      {m:8s} {lo}~{hi}")
        print()
        print("DYMOND 若是 0~N 而另外兩個是固定值，代表跑的還是舊版的 run_dymond.py。")
    else:
        print("三個模型的節點數都一致。")


if __name__ == "__main__":
    main()
