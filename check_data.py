"""檢查 nx_temporal_*.pkl 是否完整、且與 config 的宣告一致。

   python check_data.py wiki-vote
   python check_data.py wiki-vote twitter superuser

會檢查：
  1. 檔案讀得起來（pickle 沒有截斷）
  2. 結構是 list[list[nx.Graph]]
  3. 序列數 N 與序列長度 T
  4. 實測最大節點數 <= config 的 max_n
     超過的話 GNNTSampler 的 padded_x[0:n, 0:n] = x 會 broadcast 失敗，
     而且是在訓練跑了幾分鐘之後才爆。
"""
import pickle
import re
import sys

BINS = {"wiki-vote": 25600, "twitter": 200000, "superuser": 320000, "digg": 500000}


def read_cfg(ds):
    cfg = {}
    with open("experiment_configs/{}_gnn.yaml".format(ds), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = re.match(r"^  (max_n|N|T):\s*(\d+)", line)
            if m:
                cfg[m.group(1)] = int(m.group(2))
    return cfg


def check(ds):
    bins = BINS.get(ds)
    if bins is None:
        print("[{}] 未知的資料集".format(ds))
        return False

    path = "data/{}/nx_temporal_{}_{}_bins_4_timestamps_4_winlen.pkl".format(ds, ds, bins)
    print("=" * 66)
    print("{}   {}".format(ds, path))
    print("=" * 66)

    try:
        with open(path, "rb") as f:
            seqs = pickle.load(f)
    except FileNotFoundError:
        print("  [FAIL] 檔案不存在")
        return False
    except Exception as e:
        print("  [FAIL] 讀取失敗（很可能是截斷的檔案）: {}: {}".format(type(e).__name__, e))
        return False

    if not isinstance(seqs, list) or not seqs:
        print("  [FAIL] 不是非空的 list")
        return False

    lens = {len(s) for s in seqs}
    n_nodes = []
    n_edges = []
    empty = 0
    for s in seqs:
        for g in s:
            k = g.number_of_nodes()
            n_nodes.append(k)
            n_edges.append(g.number_of_edges())
            if k == 0:
                empty += 1

    cfg = read_cfg(ds)
    max_nodes = max(n_nodes)
    ok = True

    print("  序列數 N        : {:<10d} config: {}".format(len(seqs), cfg.get("N")))
    print("  序列長度 T      : {:<10s} config: {}".format(
        str(sorted(lens)) if len(lens) > 1 else str(sorted(lens)[0]), cfg.get("T")))
    print("  最大節點數      : {:<10d} config max_n: {}".format(max_nodes, cfg.get("max_n")))
    print("  平均節點 / 邊   : {:.1f} / {:.1f}".format(
        sum(n_nodes) / len(n_nodes), sum(n_edges) / len(n_edges)))
    print("  空圖數量        : {}".format(empty))

    if len(lens) > 1:
        print("  [FAIL] 序列長度不一致，DAMNETS 假設所有序列等長")
        ok = False
    elif sorted(lens)[0] != cfg.get("T"):
        print("  [WARN] T 與 config 不符")

    if cfg.get("max_n") is not None and max_nodes > cfg["max_n"]:
        print("  [FAIL] 最大節點數超過 max_n，訓練會在 padding 時 broadcast 失敗")
        print("         把 experiment_configs/{}_{{gnn,age}}.yaml 的 max_n 改成 >= {}".format(ds, max_nodes))
        ok = False

    if cfg.get("N") is not None and len(seqs) != cfg["N"]:
        print("  [WARN] N 與 config 不符（不會出錯，但要知道）")

    print("  => {}".format("OK" if ok else "有問題"))
    print()
    return ok


if __name__ == "__main__":
    targets = sys.argv[1:] or ["wiki-vote", "twitter", "superuser"]
    results = [check(d) for d in targets]
    print("=" * 66)
    print("全部通過" if all(results) else "有資料集沒通過，修好再送 job")
    sys.exit(0 if all(results) else 1)
