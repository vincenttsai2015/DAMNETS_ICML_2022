import os
import shutil
import sys

# utils 沒有 __init__.py，是靠 sys.path 找到的 namespace package，
# 解析結果會隨 CWD 改變。明確把這支檔案所在的目錄放到最前面。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import igraph
import networkx as nx
import utils.graph_utils as graph_utils
from utils.arg_helper import get_config
from baselines.DYMOND.DYMOND import get_dataset, learn_parameters, dymond_generate
from multiprocessing import Pool
from time import time

def create_directories(fstr, train_graphs):
    # 先清空再重建。收集結果時是掃遍 fstr 底下所有子目錄去讀
    # generated_graph.pklz，上次執行留下的高編號目錄沒有那個檔，
    # 序列數變少時會讀不到而中止。
    shutil.rmtree(fstr, ignore_errors=True)
    os.makedirs(fstr, exist_ok=True)
    for k, ts in enumerate(train_graphs):
        # Make a subdirectory for each timeseries to store edgelist
        ts_path = os.path.join(fstr, f'{k}')
        try:
            os.mkdir(ts_path)
        except FileExistsError:
            pass
        graph_utils.save_graph_list(ts, os.path.join(ts_path, f'nx_{k}.pkl'))


def create_dymond_datasets(fstr):
    for entry in os.scandir(fstr):
        ts = graph_utils.load_graph_ts(os.path.join(entry.path, f'nx_{entry.name}.pkl'))
        edgelist = []
        edge_timesteps = []
        for t, G in enumerate(ts):
            edgelist += list(G.edges)
            edge_timesteps += [t + 1 for _ in G.edges]
        g = igraph.Graph(n=G.number_of_nodes(), directed=False,
                         edges=edgelist, edge_attrs={'timestep': edge_timesteps})
        for v in g.vs:
            v['nid'] = f'nid-{v.index}'  # Annotate with original index
            neighbors = list(set([u for u in g.neighbors(v)]))
            if len(neighbors) > 0:
                v_edges = g.es.select(_between=([v.index], neighbors))
                v['active'] = min(v_edges['timestep'])

        # Save to file
        graph_filename = f'{entry.name}_ig.pklz'
        g.write_picklez(os.path.join(entry.path, graph_filename))

        T = len(ts)
        timesteps = [i + 1 for i in range(T)]
        dataset_info = {'gname': graph_filename,
                        'L': 1,
                        'N': g.vcount(),
                        'T': len(timesteps),
                        'timesteps': timesteps
                        }
        dataset_info_file = os.path.join(entry.path, 'dataset_info.pkl')
        with open(dataset_info_file, 'wb') as output:
            pickle.dump(dataset_info, output)


def train_test_dymond(path):
    # for entry in os.scandir(fstr):
    dataset_dir, dataset_info, g = get_dataset(dataset_dir=path)
    learn_parameters(dataset_dir, dataset_info, g)
    dymond_generate(dataset_dir, dataset_info['T'] + 1)

def run_dymond(test_dir=None, graphs_file=None):
    start_time = time()
    if test_dir is None:
        with open('experiment_files/last_test.txt', 'r') as f:
            test_dir = f.readline()
        args = get_config(os.path.join(test_dir, 'config.yaml'))
        test_dir = args.experiment.test.graph_dir

    if graphs_file is None:
        graphs_file = 'test_graphs.pkl'
    train_graphs = graph_utils.load_graph_ts(os.path.join(test_dir, graphs_file))
    T = len(train_graphs[0])
    fstr = os.path.join(test_dir, 'train_edgelists')

    create_directories(fstr, train_graphs)
    create_dymond_datasets(fstr)
    dirs = [entry.path for entry in os.scandir(fstr)]
    if len(dirs) == 1:
        train_test_dymond(dirs[0])
    else:
        # Pool() 預設用 os.cpu_count()，那是整台機器的核數；
        # SLURM 只配給我們 --cpus-per-task 個，用 affinity 才拿得到正確數字。
        n_proc = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
        with Pool(processes=n_proc) as p:
            p.map(train_test_dymond, dirs)
    ts_list = []

    # DAMNETS 與 AGE 都把每張補到 config 的 max_n，也就是資料裡的最大節點數。
    # DYMOND 若只補到各自來源序列的節點數，零度節點的數量會不同，
    # degree 直方圖因此偏移，MMD 比較就不公平。
    max_n = max(g.number_of_nodes() for s in train_graphs for g in s)

    # scandir 的順序不保證是數字序，而目錄名就是 train_graphs 的索引。
    # 排序之後生成序列才與 test_graphs.pkl 一一對應。
    entries = sorted(os.scandir(fstr), key=lambda e: int(e.name))
    for entry in entries:
        sampled_ts = []
        sampled_fstr = os.path.join(entry.path, 'learned_parameters/generated_graph/generated_graph.pklz')
        ig_ts = igraph.Graph().Read_Picklez(sampled_fstr)
        # from_edgelist 只建立有邊的節點，孤立節點會消失、整張沒邊就成了空圖。
        # DAMNETS 與 AGE 是從固定大小的鄰接矩陣還原、節點集完整，
        # 節點數不一致會讓 degree 分佈的直方圖退化，MMD 算出負值。
        n_nodes = max(max_n, ig_ts.vcount())
        for t in range(1, T + 1):
            g = nx.Graph()
            g.add_nodes_from(range(n_nodes))
            g.add_edges_from(e.tuple for e in ig_ts.es.select(lambda e: e['timestep'] == t))
            sampled_ts.append(g)
        ts_list.append(sampled_ts)
    end_time = time() - start_time
    print(f'Training time: {end_time}')
    graph_utils.save_graph_list(ts_list, os.path.join(test_dir, 'sampled_ts.pkl'))
    # 每條序列都會留下 igraph pickle 與 learned_parameters，結果收進
    # sampled_ts.pkl 之後就沒用了。累積下來會把家目錄配額塞爆。
    shutil.rmtree(fstr, ignore_errors=True)

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    test_dir = f'../test_and_generated_graphs/{dataset_name}/DYMOND'
    run_dymond(test_dir, 'test_graphs.pkl')



