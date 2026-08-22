import networkx as nx
import pickle
import numpy as np
import os
import shutil
import utils.graph_utils as graph_utils
from utils.arg_helper import get_config
from types import SimpleNamespace
from baselines.TagGen.graph_fairnet import Config, main, data_process
from time import time

def create_edgelists(el_fstr, train_graphs):
    '''
    The TagGen code requires a certain filestructure to work, namely a data directory that contains a timestamped
    edgelist for each network. So this function just creates this file structure.
    Args:
        test_dir: The directory of the test run (of the graph encoder model) that you want to run the baselines on.
        The sampled graphs from tagGen will be placed there
    Returns: None

    '''
    # 先清掉整個目錄再重建。_tag_gen_one 跑完會刪掉自己那一層，中途失敗時
    # 已完成的部分不見、未完成的留著；下次執行若序列數較少，殘留的高編號目錄
    # 會被 os.scandir 掃進來，而它的 edgelist.txt 早就被刪了。
    shutil.rmtree(el_fstr, ignore_errors=True)
    os.makedirs(el_fstr, exist_ok=True)
    # Loop through each time series in training set
    for k, ts in enumerate(train_graphs):
        # Make a subdirectory for each timeseries to store edgelist
        ts_path = os.path.join(el_fstr, f'{k}')
        try:
            os.mkdir(ts_path)
        except FileExistsError:
            pass
        # Write each graph into an edgelist file (one per time series)
        with open(os.path.join(ts_path, 'edgelist.txt'), 'w') as f:
            for t, g in enumerate(ts):
                for i, j in list(g.edges()):
                    f.write(f'{i} {j} {t}\n')
        graph_utils.save_graph_list(ts, os.path.join(ts_path, f'nx_{k}.pkl'))

def _tag_gen_one(job):
    """單一條序列的完整 TagGen 流程。抽成模組層級的函式才能給 multiprocessing 用。"""
    path, name, T = job
    tg_args = SimpleNamespace()
    tg_args.slices = T
    tg_args.window = 1
    tg_args.gpu = 0
    tg_args.biased = True

    config = Config()
    tg_args.data_path = os.path.join(path, 'sequences.txt')
    tg_args.embedding = os.path.join(path, f'{name}_emb')
    config.embedding = tg_args.embedding
    config.node_embedding = os.path.join(path, f'{name}_node_level_emb')
    config.use_output_path = os.path.join(path, f'{name}_output_sequences.txt')
    tg_args.emb_size = config.d_model
    tg_args.data = name

    tg_args.mode = True
    # preprocess_edgelist 算的是 int((最大時間戳 - 最小時間戳 + 1) / interval)，
    # 為 0 就中止。序列頭尾若有沒有邊的 snapshot，實際跨度會小於宣告的 T，
    # 所以 interval 取實際跨度而不是 T。
    edgelist = os.path.join(path, 'edgelist.txt')
    stamps = set()
    with open(edgelist) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                stamps.add(int(parts[2]))
    span = (max(stamps) - min(stamps) + 1) if stamps else 0
    interval = min(T, span) if span else T

    data_process(tg_args, interval, tg_args.biased, time_windows=tg_args.window,
                 data_directory=edgelist,
                 output_directory=path, directed=False)
    main(tg_args, config, path)
    tg_args.mode = False
    main(tg_args, config, path)

    # 每條序列會留下 edgelist.txt、sequences.txt、兩份 word2vec embedding、
    # output_sequences.txt、graph.pickle、edgelist_new.txt。原本全部留到最後才一起讀，
    # 序列數上萬時會把家目錄配額塞爆（Errno 122）。改成跑完立刻取出結果並刪掉目錄，
    # 暫存量因此固定在同時執行的 worker 數，而不是序列總數。
    ts_array = load_tag_gen_results(path, config.use_output_path)
    # ts_array 的張數由縮放後的時間戳範圍決定，序列頭尾沒有邊時會少於 T。
    # 評估要求每條序列都是 T 張，不足的補空圖。
    n_nodes = ts_array[0].shape[0] if len(ts_array) else 0
    graphs = []
    for t in range(T):
        if t < len(ts_array):
            graphs.append(nx.Graph(ts_array[t]))
        else:
            g = nx.Graph()
            g.add_nodes_from(range(n_nodes))
            graphs.append(g)
    shutil.rmtree(path, ignore_errors=True)
    return int(name), graphs


def train_test_tag_gen(T, el_fstr):
    jobs = sorted(((e.path, e.name, T) for e in os.scandir(el_fstr)),
                  key=lambda j: int(j[1]))
    # 每條序列都要完整訓練一次 transformer，序列數大時單執行緒跑不完。
    # TAGGEN_WORKERS 控制平行度，1 等同原本的序列執行。
    # 各條序列各自寫入自己的目錄、彼此不共用狀態，用 spawn 讓每個行程有獨立的
    # Config 與 CUDA context。
    workers = int(os.environ.get('TAGGEN_WORKERS', '1'))
    print(f'[TagGen] {len(jobs)} sequences, workers={workers}', flush=True)

    results = {}
    if workers <= 1:
        for i, j in enumerate(jobs, 1):
            idx, graphs = _tag_gen_one(j)
            results[idx] = graphs
            if i % 50 == 0 or i == len(jobs):
                print(f'[TagGen] {i}/{len(jobs)} done', flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=workers) as pool:
            for i, (idx, graphs) in enumerate(pool.imap_unordered(_tag_gen_one, jobs), 1):
                results[idx] = graphs
                if i % 50 == 0 or i == len(jobs):
                    print(f'[TagGen] {i}/{len(jobs)} done', flush=True)

    # 依原始序列編號排序，讓輸出與 test_graphs 的順序一致（KS 是逐序列配對的）
    return [results[k] for k in sorted(results)]


def dictionary_search(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key

def load_tag_gen_results(output_dir, data_directory_2):
    graph_attr = pickle.load(open(output_dir + '/graph.pickle', "rb"))
    original_network = graph_attr['graph']
    original_node_index = graph_attr['original_index']
    node_index = graph_attr['index']
    n = original_network.shape[0]
    min_time_stamp = np.inf
    max_time_stamp = 0
    with open(output_dir + '/edgelist_new.txt', 'r') as f:
        for line in f:
            line = list(map(int, line.split()))
            if line[2] < min_time_stamp:
                min_time_stamp = line[2]
            if line[2] > max_time_stamp:
                max_time_stamp = line[2]
    windows = max_time_stamp - min_time_stamp + 1
    original_network = np.zeros((windows, n, n), dtype=np.int8)
    with open(output_dir + '/edgelist_new.txt', 'r') as f:
        for line in f:
            line = list(map(int, line.split()))
            a_1 = int(dictionary_search(node_index, line[0]).split('_')[0])
            a_2 = int(dictionary_search(node_index, line[1]).split('_')[0])
            index_i = original_node_index[a_1]
            index_j = original_node_index[a_2]
            for k in range(line[2], max_time_stamp + 1):
                original_network[k, index_i, index_j] = 1
                original_network[k, index_j, index_i] = 1
    for i in range(n):
        for k in range(windows):
            original_network[k, i, i] = 1
    graph = np.zeros((windows, n, n), dtype=np.float16)
    edge_count = [int(np.sum(original_network[k])) for k in range(windows)]
    with open(data_directory_2, 'r+') as f:
        for line in f:
            line = line.rstrip("\n")
            # 取樣寫出的序列檔可能有空行或結尾換行，int('') 會中止整條序列
            if not line.strip():
                continue
            nodes = [int(x) for x in line.split(',') if x.strip()]
            if len(nodes) < 2:
                continue
            for i in range(len(nodes) - 1):
                if i <= len(nodes) - 1:
                    a_1 = list(map(int, dictionary_search(node_index, nodes[i]).split('_')))
                    a_2 = list(map(int, dictionary_search(node_index, nodes[i+1]).split('_')))
                    time_stamp = max(a_1[1], a_2[1])
                    index_i = original_node_index[a_1[0]]
                    index_j = original_node_index[a_2[0]]
                    r = np.random.uniform(low=0.85, high=1)
                    for k in range(time_stamp, windows):
                        graph[k, index_i, index_j] += r
                        graph[k, index_j, index_i] += r
    for i in range(n):
        for k in range(windows):
            graph[k, i, i] = graph[k, i, i] + np.random.uniform(low=0.85, high=1)
    for k in range(windows):
        DD = np.sort(graph[k].flatten())[::-1]
        threshold = DD[edge_count[k]]
        graph[k] = np.array(
            [[0 if graph[k, i, j] <= threshold else 1 for i in range(graph.shape[1])]
             for j in range(graph.shape[2])], dtype=np.int8)
    return graph

def run_tag_gen(test_dir=None, graphs_file=None):
    if test_dir is None:
        with open('experiment_files/last_test.txt', 'r') as f:
            test_dir = f.readline()
            args = get_config(os.path.join(test_dir, 'config.yaml'))
            test_dir = args.experiment.test.graph_dir
    if graphs_file is None:
        graphs_file = 'test_graphs.pkl'
    # 原本寫死 [-5:]，只取最後 5 條序列。TAGGEN_MAX_SEQS 控制取幾條，
    # 預設 0 表示使用完整的 test 集，與其他三個模型一致。
    train_graphs = graph_utils.load_graph_ts(os.path.join(test_dir, graphs_file))
    _cap = int(os.environ.get("TAGGEN_MAX_SEQS", "0"))
    if _cap:
        train_graphs = train_graphs[-_cap:]
    print(f"[TagGen] using {len(train_graphs)} sequences "
          f"(TAGGEN_MAX_SEQS={_cap or 'all'})", flush=True)
    T = len(train_graphs[0])
    el_fstr = os.path.join(test_dir, 'train_edgelists')
    create_edgelists(el_fstr, train_graphs)
    train_start_time = time()
    ts_list = train_test_tag_gen(T, el_fstr)
    train_end_time = time()-train_start_time
    print(f'Training time: {train_end_time}')
    print(f'[TagGen] collected {len(ts_list)} sequences', flush=True)

    shutil.rmtree(el_fstr, ignore_errors=True)
    graph_utils.save_graph_list(ts_list, os.path.join(test_dir, 'sampled_ts.pkl'))

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    test_dir = f'../test_and_generated_graphs/{dataset_name}/TagGen'
    run_tag_gen(test_dir, 'test_graphs.pkl')