import networkx as nx
import numpy as np
import pickle
from utils.ba_ts_generator import barabasi_albert_graph_ts
from functools import partial
from multiprocessing import Pool


def generate_graphs(args):
    graph_type = args.name
    if graph_type == '3_comm_decay':
        fn = n_community_decay_ts
    elif graph_type == '3_comm_const':
        fn = n_community_const_ts
    # Used for testing models
    elif graph_type == '3_comm_interpolation':
        g = n_community_const_ts(c_sizes=args.c_sizes, T=args.T, p=args.p_int, p_ext=args.p_ext)
        train_list = [g for _ in range(args.N)]
        test_list = [g for _ in range(args.N)]
    elif graph_type == 'ba':
        fn = barabasi_albert_graph_ts
    elif graph_type == 'gowalla':
        train_list, test_list = gowalla_loader(path=args.path, ratio=args.ratio)

    if graph_type != '3_comm_interpolation' or graph_type != 'gowalla':
        fn = partial(fn, **args)
        fn = partial(wrapper, fn=fn)
        with Pool(args.n_workers) as p:
            train_list = p.map(fn, [_ for _ in range(args.N)])
            test_list = p.map(fn, [_ for _ in range(args.N)])
    args.T = len(train_list[0])

    return train_list, test_list


def wrapper(x, fn):
    return fn()


def gowalla_loader(path, ratio=0.8):
    with open(path,'rb') as f:
        time_seqs = pickle.load(f)    
    train_list, test_list = time_seqs[0:int(ratio*len(time_seqs))], time_seqs[int(ratio*len(time_seqs)):]
    return train_list, test_list

def n_community_decay_ts(c_sizes, T, p_int=0.7, p_ext=0.01, decay_prop=0.2, **kwargs):
    G = nx.random_partition_graph(c_sizes, p_int, p_ext)
    communities = [G.subgraph(c) for c in G.graph['partition']]
    decay_edges = list(communities[-1].edges)
    decay_nodes = list(communities[-1].nodes)
    non_decay_nodes = [node for comm in communities[:-1] for node in comm]
    # G = add_external_connections(G, communities, p_ext)

    ts = [G]
    for t in range(1, T):
        G_t = nx.Graph(ts[t - 1])
        # Change a proportion of 'in' edges to 'out' edges in the decay community
        n = int(decay_prop * len(decay_edges))
        for i in range(n):
            ix = np.random.choice(len(decay_edges))
            edge = decay_edges.pop(ix)
            G_t.remove_edge(*edge)
            # Add back the same number of random 'out' edges
            edge_exists = True
            while edge_exists:
                n1 = np.random.choice(decay_nodes)
                n2 = np.random.choice(non_decay_nodes)
                edge_exists = G_t.has_edge(n1, n2)
            G_t.add_edge(n1, n2)
        ts.append(G_t)
    return ts


def n_community_const_ts(c_sizes, T, p_int=0.7, p_ext=0.01, **kwargs):
    # Used for model verification
    ts = []
    G = nx.random_partition_graph(c_sizes, p_int, p_ext)
    for t in range(T):
        ts.append(G)
    return ts