import graph_tool.all as gt
##Navigate to the ./util/orca directory and compile orca.cpp
# g++ -O2 -std=c++11 -o orca orca.cpp
import os, sys
import copy
import pickle
import random
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

import pygsp as pg
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from scipy.linalg import eigvalsh
from scipy.stats import chi2, ks_2samp
from dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv

PRINT_TIME = False
__all__ = ['degree_stats', 'clustering_stats', 'orbit_stats_all', 'spectral_stats', 'eval_acc_lobster_graph']

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(
                nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

###############################################################################
def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf

def get_spectral_pmf(eigs, max_eig):
    spectral_pmf, _ = np.histogram(np.clip(eigs, 0, max_eig), bins=200, range=(-1e-5, max_eig), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf

def eigval_stats(eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_pmf, eig_ref_list,
                                                 [max_eig for i in range(len(eig_ref_list))]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_pmf, eig_pred_list,
                                                 [max_eig for i in range(len(eig_ref_list))]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eig_ref_list)):
            spectral_temp = get_spectral_pmf(eig_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(eig_pred_list)):
            spectral_temp = get_spectral_pmf(eig_pred_list[i])
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing eig mmd: ', elapsed)
    return mmd_dist

def eigh_worker(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)

def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list

def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop ** 2, axis=2)
    hist_range = [0, bound]
    hist = np.array([np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt])  # NOTE: change number of bins
    return hist.flatten()

def spectral_filter_stats(eigvec_ref_list, eigval_ref_list, eigvec_pred_list, eigval_pred_list, is_parallel=False,
                          compute_emd=False):
    ''' Compute the distance between the eigvector sets.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""
        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_filter_worker, eigvec_ref_list, eigval_ref_list,
                                                 [filters for i in range(len(eigval_ref_list))],
                                                 [bound for i in range(len(eigval_ref_list))]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(get_spectral_filter_worker, eigvec_pred_list, eigval_pred_list,
                                                 [filters for i in range(len(eigval_ref_list))],
                                                 [bound for i in range(len(eigval_ref_list))]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(eigvec_ref_list[i], eigval_ref_list[i], filters, bound)
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(eigvec_pred_list[i], eigval_pred_list[i], filters, bound)
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing spectral filter stats: ', elapsed)
    return mmd_dist

def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty,
                                                 [n_eigvals for i in graph_ref_list]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

###############################################################################
def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_stats(graph_ref_list,
                     graph_pred_list,
                     bins=100,
                     is_parallel=True, compute_emd=False):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                    clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)

        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist

# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts:'

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

def orca(graph):
    # tmp_fname = f'analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = f'orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    # print(tmp_fname, flush=True)
    f = open(tmp_fname, 'w')
    f.write(
        str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()
    output = sp.check_output([str(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'orca/orca')), 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.split(' '))) for node_cnts in output.strip('\n').split('\n')]) # there is a value error occurred here!

    try:
        os.remove(tmp_fname)
    except OSError:
        pass
    return node_orbit_counts

def motif_stats(graph_ref_list, graph_pred_list, motif_type='4cycle', ground_truth_match=None, bins=100, compute_emd=False):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_ref = np.array(total_counts_ref)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, is_hist=False)
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False)
    else:
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False)
    return mmd_dist

def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    # mmd_dist = compute_mmd(
    #     total_counts_ref,
    #     total_counts_pred,
    #     kernel=gaussian,
    #     is_hist=False,
    #     sigma=30.0)

    # mmd_dist = compute_mmd(
    #         total_counts_ref,
    #         total_counts_pred,
    #         kernel=gaussian_tv,
    #         is_hist=False,
    #         sigma=30.0)
    if compute_emd:
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, sigma=30.0)
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)
    else:
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv, is_hist=False, sigma=30.0)
    return mmd_dist

def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1
    return count / float(len(G_list))

def eval_acc_tree_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_tree(gg):
            count += 1
    return count / float(len(G_list))

def eval_acc_grid_graph(G_list, grid_start=10, grid_end=20):
    count = 0
    for gg in G_list:
        if is_grid_graph(gg):
            count += 1
    return count / float(len(G_list))

def eval_acc_sbm_graph(G_list, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000, is_parallel=True):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(is_sbm_graph,
                                     [gg for gg in G_list], [p_intra for i in range(len(G_list))],
                                     [p_inter for i in range(len(G_list))],
                                     [strict for i in range(len(G_list))],
                                     [refinement_steps for i in range(len(G_list))]):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(gg, p_intra=p_intra, p_inter=p_inter, strict=strict,
                                  refinement_steps=refinement_steps)
    return count / float(len(G_list))

def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))

def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]

def is_lobster_graph(G):
    """
        Check a given graph is a lobster graph or not

        Removing leaf nodes twice:

        lobster -> caterpillar -> path

    """
    ### Check if G is a tree
    if nx.is_tree(G):
        G = G.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False

def is_grid_graph(G):
    """
    Check if the graph is grid, by comparing with all the real grids with the same node count
    """
    all_grid_file = f"data/all_grids.pt"
    if os.path.isfile(all_grid_file):
        all_grids = torch.load(all_grid_file)
    else:
        all_grids = {}
        for i in range(2, 20):
            for j in range(2, 20):
                G_grid = nx.grid_2d_graph(i, j)
                n_nodes = f"{len(G_grid.nodes())}"
                all_grids[n_nodes] = all_grids.get(n_nodes, []) + [G_grid]
        torch.save(all_grids, all_grid_file)

    n_nodes = f"{len(G.nodes())}"
    if n_nodes in all_grids:
        for G_grid in all_grids[n_nodes]:
            if nx.faster_could_be_isomorphic(G, G_grid):
                if nx.is_isomorphic(G, G_grid):
                    return True
        return False
    else:
        return False

def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """
    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9  # p value < 10 %
    else:
        return p

def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))

def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs
    return frac_unique

def eval_fraction_unique_non_isomorphic_valid(fake_graphs, train_graphs, validity_func=(lambda x: True)):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (float(len(fake_graphs)) - count_non_unique - count_isomorphic) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid

# -------- Node Behavior Metric (2D KS using Q1 and Q3 of degree) -------- #
def extract_q1_q3_points(graph_seq):
    q1_q3_points = []
    # print(f'graph_seq: {graph_seq}')
    for g in graph_seq:
        # print(f'g: {g}')
        # print(f'g.nodes: {g.nodes()}')
        # print(f'g.edges: {g.edges()}')
        degrees = np.array([d for _, d in g.degree()])
        if len(degrees) < 4:
            continue
        q1 = np.percentile(degrees, 25)
        q3 = np.percentile(degrees, 75)
        q1_q3_points.append([q1, q3])
    return np.array(q1_q3_points)

def compute_2d_ks(test_points, gen_points):
    ks_x = ks_2samp(test_points[:, 0], gen_points[:, 0]).statistic
    ks_y = ks_2samp(test_points[:, 1], gen_points[:, 1]).statistic
    return (ks_x + ks_y) / 2

def run_node_behavior_eval(test_graph_seqs, gen_graph_seqs):
    # print(test_graphs[0])
    # print(gen_graphs[0])
    n = min(len(test_graph_seqs), len(gen_graph_seqs))
    ks_scores = []
    for i in range(n):
        # print(f'test_graphs[{i}]: {test_graphs[i]}')
        test_points = extract_q1_q3_points(test_graph_seqs[i])
        gen_points = extract_q1_q3_points(gen_graph_seqs[i])
        if len(test_points) == 0 or len(gen_points) == 0:
            continue
        ks = compute_2d_ks(test_points, gen_points)
        ks_scores.append(ks)
    return np.mean(ks_scores) if ks_scores else None

# -------- Dynamical Similarity via Random Walk Coverage -------- #
def random_walk_coverage(graph_seq, walk_times=10):
    T = len(graph_seq)
    cover_counts = []
    for _ in range(walk_times):
        visited = set()
        g0 = graph_seq[0]
        if len(g0.nodes) == 0:
            continue
        current = random.choice(list(g0.nodes))
        visited.add(current)
        for t in range(1, T):
            g = graph_seq[t]
            if current not in g:
                break
            neighbors = list(g.neighbors(current))
            if neighbors:
                current = random.choice(neighbors)
            visited.add(current)
        cover_counts.append(len(visited))
    return cover_counts

def run_dynamic_sim_eval(test_graph_seqs, gen_graph_seqs):
    n = min(len(test_graph_seqs), len(gen_graph_seqs))
    test_cover, gen_cover = [], []
    for i in range(n):
        test_cover += random_walk_coverage(test_graph_seqs[i])
        gen_cover += random_walk_coverage(gen_graph_seqs[i])
    if not test_cover or not gen_cover:
        return None
    return ks_2samp(test_cover, gen_cover).statistic

def MMD_evaluator(generated_graph_snaoshots, reference_graph_snaoshots):
    num_timestamps = len(reference_graph_snaoshots)
    print('Computing MMD of degree distributions...')
    degree_snapshots = np.array([degree_stats(reference_graph_snaoshots[t], generated_graph_snaoshots[t], is_parallel=True, compute_emd=False) for t in range(num_timestamps)])
    avg_degree_MMD = np.mean(degree_snapshots)
    print('Computing MMD of clustering coefficient distributions...')
    clustering_snapshots = np.array([clustering_stats(reference_graph_snaoshots[t], generated_graph_snaoshots[t], is_parallel=True, compute_emd=False) for t in range(num_timestamps)])
    avg_clustering_MMD = np.mean(clustering_snapshots)
    print('Computing MMD of spectral distributions...')
    spectral_snapshots = np.array([spectral_stats(reference_graph_snaoshots[t], generated_graph_snaoshots[t], is_parallel=True, compute_emd=False) for t in range(num_timestamps)])
    avg_spectral_MMD = np.mean(spectral_snapshots)
    
    return avg_degree_MMD, avg_clustering_MMD, avg_spectral_MMD

def KS_evaluator(generated_graph_seqs, reference_graph_seqs):
    print('Computing node behavior KS...')
    node_bahavior_ks = run_node_behavior_eval(generated_graph_seqs, reference_graph_seqs)
    print('Computing RW KS...')
    random_walk_ks = run_dynamic_sim_eval(generated_graph_seqs, reference_graph_seqs)
    return node_bahavior_ks, random_walk_ks


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    model_list = ['DYMOND','AGE','DAMNET','TagGen']
    for model_name in model_list:
        print('Loading testing and sampled graph data...')
        test_graph_path = f'../test_and_generated_graphs/{dataset_name}/{model_name}/test_graphs.pkl' # fill in the corresponding info
        sampled_graph_path = f'../test_and_generated_graphs/{dataset_name}/{model_name}/sampled_ts.pkl' # fill in the corresponding info
        with open(sampled_graph_path,'rb') as f_gen:
            sampled_graphs = pickle.load(f_gen)
            print(f'There are {len(sampled_graphs)} generated temporal graph sequences of length {len(sampled_graphs[0])}.')
        with open(test_graph_path,'rb') as f_test:
            test_graphs = pickle.load(f_test)
            print(f'There are {len(test_graphs)} testing temporal graph sequences of length {len(test_graphs[0])}.')
        
        print('Transposing the data structure...')
        transposed_sampled_graphs = list(map(list, zip(*sampled_graphs)))
        transposed_test_graphs = list(map(list, zip(*test_graphs)))

        print('Computing evaluation metrics...')
        num_seqs = min(len(test_graphs), len(sampled_graphs))
        print('Computing MMD...')
        avg_degree_MMD, avg_clustering_MMD, avg_spectral_MMD = MMD_evaluator(transposed_sampled_graphs, transposed_test_graphs)
        print('Computing KS...')
        node_bahavior_ks, random_walk_ks = KS_evaluator(sampled_graphs[0:num_seqs], test_graphs[0:num_seqs])
        print(f'Average degree MMD in data generated by {model_name}: {avg_degree_MMD}')
        print(f'Average clustering MMD in data generated by {model_name}: {avg_clustering_MMD}')
        print(f'Average spectral MMD in data generated by {model_name}: {avg_spectral_MMD}')
        print(f'Node behavior KS in data generated by {model_name}: {node_bahavior_ks}')
        print(f'Random walk KS in data generated by {model_name}: {random_walk_ks}')