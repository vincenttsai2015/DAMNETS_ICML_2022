import numpy as np
import networkx as nx
import concurrent.futures
from datetime import datetime
import pickle
from scipy.linalg import eigvalsh
import os,sys
import pygsp as pg
import secrets
from string import ascii_uppercase, digits
from scipy.stats import chi2
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_networkx

# Note: Ensure dist_helper.py is in the same directory or Python path
try:
    from dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv, disc
except ImportError:
    raise ImportError("dist_helper module not found. Ensure dist_helper.py is in /home/u3797948/DAMNETS_ICML_2022/models/ or in the Python path.")

PRINT_TIME = False

# --- Statistical Functions ---

def degree_worker(G):
    """Compute degree histogram for a graph."""
    return np.array(nx.degree_histogram(G))

def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    """Compute MMD between degree distributions of reference and generated graphs."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if G.number_of_nodes() > 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for G in graph_ref_list:
            sample_ref.append(degree_worker(G))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(degree_worker(G))

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

def clustering_worker(param):
    """Compute clustering coefficient histogram for a graph."""
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False):
    """Compute MMD between clustering coefficient distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if G.number_of_nodes() > 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for G in graph_ref_list:
            sample_ref.append(clustering_worker((G, bins)))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(clustering_worker((G, bins)))

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist

def spectral_worker(G, n_eigvals=-1):
    """Compute spectral histogram from normalized Laplacian eigenvalues."""
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except Exception as e:
        print(f"Error in spectral_worker for graph with {G.number_of_nodes()} nodes: {e}")
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum() if spectral_pmf.sum() > 0 else spectral_pmf
    return spectral_pmf

def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False):
    """Compute MMD between spectral distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if G.number_of_nodes() > 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list, [n_eigvals for _ in graph_ref_list]):
                sample_ref.append(spectral_density)
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty, [n_eigvals for _ in graph_pred_list_remove_empty]):
                sample_pred.append(spectral_density)
    else:
        for G in graph_ref_list:
            sample_ref.append(spectral_worker(G, n_eigvals))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(spectral_worker(G, n_eigvals))

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing spectral mmd: ', elapsed)
    return mmd_dist

# --- Distribution and Evaluation Functions ---

def compute_distributions(graph_list, is_parallel=True, bins=100, n_eigvals=-1):
    """Compute degree, clustering, and spectral distributions for a list of graphs."""
    degree_hists = []
    clustering_hists = []
    spectral_hists = []
    
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            degree_hists = list(executor.map(degree_worker, graph_list))
            clustering_hists = list(executor.map(clustering_worker, [(G, bins) for G in graph_list]))
            spectral_hists = list(executor.map(spectral_worker, graph_list, [n_eigvals for _ in graph_list]))
    else:
        for G in graph_list:
            degree_hists.append(degree_worker(G))
            clustering_hists.append(clustering_worker((G, bins)))
            spectral_hists.append(spectral_worker(G, n_eigvals))
    
    return degree_hists, clustering_hists, spectral_hists

def evaluator_detailed(generated_graphs, reference_graphs, is_parallel=True, compute_emd=False, bins=100, n_eigvals=-1):
    """Compute MMD and distributions for degree, clustering, and spectral stats at each timestamp."""
    num_timestamps = len(reference_graphs)
    if len(generated_graphs) != num_timestamps:
        raise ValueError(f"Mismatch in number of timestamps: {len(generated_graphs)} vs {num_timestamps}")
    
    results = {
        'degree_mmd': [],
        'clustering_mmd': [],
        'spectral_mmd': [],
        'degree_ref_dist': [],
        'degree_gen_dist': [],
        'clustering_ref_dist': [],
        'clustering_gen_dist': [],
        'spectral_ref_dist': [],
        'spectral_gen_dist': []
    }
    
    for t in range(num_timestamps):
        ref_graphs = reference_graphs[t]
        gen_graphs = [G for G in generated_graphs[t] if G.number_of_nodes() > 0]
        
        empty_count = len(generated_graphs[t]) - len(gen_graphs)
        if empty_count > 0:
            print(f"Timestamp {t}: Removed {empty_count} empty graphs from generated set.")
        
        try:
            degree_mmd = degree_stats(ref_graphs, gen_graphs, is_parallel=is_parallel, compute_emd=compute_emd)
            clustering_mmd = clustering_stats(ref_graphs, gen_graphs, bins=bins, is_parallel=is_parallel, compute_emd=compute_emd)
            spectral_mmd = spectral_stats(ref_graphs, gen_graphs, is_parallel=is_parallel, n_eigvals=n_eigvals, compute_emd=compute_emd)
        except Exception as e:
            print(f"Error computing MMD at timestamp {t}: {e}")
            degree_mmd = clustering_mmd = spectral_mmd = float('inf')
        
        ref_degree_hists, ref_clustering_hists, ref_spectral_hists = compute_distributions(ref_graphs, is_parallel, bins, n_eigvals)
        gen_degree_hists, gen_clustering_hists, gen_spectral_hists = compute_distributions(gen_graphs, is_parallel, bins, n_eigvals)
        
        results['degree_mmd'].append(degree_mmd)
        results['clustering_mmd'].append(clustering_mmd)
        results['spectral_mmd'].append(spectral_mmd)
        results['degree_ref_dist'].append(ref_degree_hists)
        results['degree_gen_dist'].append(gen_degree_hists)
        results['clustering_ref_dist'].append(ref_clustering_hists)
        results['clustering_gen_dist'].append(gen_clustering_hists)
        results['spectral_ref_dist'].append(ref_spectral_hists)
        results['spectral_gen_dist'].append(gen_spectral_hists)
    
    results['avg_degree_mmd'] = float(sum([x for x in results['degree_mmd'] if x != float('inf')]) / len([x for x in results['degree_mmd'] if x != float('inf')])) if any(x != float('inf') for x in results['degree_mmd']) else float('inf')
    results['avg_clustering_mmd'] = float(sum([x for x in results['clustering_mmd'] if x != float('inf')]) / len([x for x in results['clustering_mmd'] if x != float('inf')])) if any(x != float('inf') for x in results['clustering_mmd']) else float('inf')
    results['avg_spectral_mmd'] = float(sum([x for x in results['spectral_mmd'] if x != float('inf')]) / len([x for x in results['spectral_mmd'] if x != float('inf')])) if any(x != float('inf') for x in results['spectral_mmd']) else float('inf')
    
    return results

# --- Main Execution ---

if __name__ == '__main__':
    dataset_list = ['digg','twitter','wiki-vote','superuser']
    model_list = ['AGE','DAMNET','TagGen']
    for dataset_name in dataset_list:
        for model_name in model_list:
            test_graph_path = f'/home/u3797948/test_and_generated_graphs/{dataset_name}/{model_name}/test_graphs.pkl'
            sampled_graph_path = f'/home/u3797948/test_and_generated_graphs/{dataset_name}/{model_name}/sampled_ts.pkl'
            
            try:
                # Verify file existence
                if not os.path.exists(test_graph_path):
                    raise FileNotFoundError(f"Test graph file not found: {test_graph_path}")
                if not os.path.exists(sampled_graph_path):
                    raise FileNotFoundError(f"Sampled graph file not found: {sampled_graph_path}")

                with open(sampled_graph_path, 'rb') as f_gen:
                    sampled_graphs = pickle.load(f_gen)
                    print(f'There are {len(sampled_graphs)} generated temporal graph sequences of length {len(sampled_graphs[0])}.')
                with open(test_graph_path, 'rb') as f_test:
                    test_graphs = pickle.load(f_test)
                    print(f'There are {len(test_graphs)} testing temporal graph sequences of length {len(test_graphs[0])}.')
            
                reference_graphs = [[] for _ in range(len(test_graphs[0]))]
                generated_graphs = [[] for _ in range(len(sampled_graphs[0]))]
                for seq in test_graphs:
                    for t in range(len(seq)):
                        reference_graphs[t].append(seq[t])
                for seq in sampled_graphs:
                    for t in range(len(seq)):
                        generated_graphs[t].append(seq[t])
            
                results = evaluator_detailed(generated_graphs, reference_graphs, is_parallel=True, compute_emd=False, bins=100, n_eigvals=-1)
            
                print("\nPer-timestamp MMD scores:")
                for t in range(len(results['degree_mmd'])):
                    print(f"Timestamp {t}:")
                    print(f"  Degree MMD: {results['degree_mmd'][t]:.6f}")
                    print(f"  Clustering MMD: {results['clustering_mmd'][t]:.6f}")
                    print(f"  Spectral MMD: {results['spectral_mmd'][t]:.6f}")
            
                print("\nAverage MMD scores:")
                print(f"  Average Degree MMD: {results['avg_degree_mmd']:.6f}")
                print(f"  Average Clustering MMD: {results['avg_clustering_mmd']:.6f}")
                print(f"  Average Spectral MMD: {results['avg_spectral_mmd']:.6f}")
            
                # Pad histograms to the maximum length
                max_len_ref = max(len(hist) for hist in results['degree_ref_dist'][0])
                max_len_gen = max(len(hist) for hist in results['degree_gen_dist'][0])
                max_len = max(max_len_ref, max_len_gen)
                
                padded_ref_hists = [np.pad(hist, (0, max_len - len(hist)), mode='constant') for hist in results['degree_ref_dist'][0]]
                padded_gen_hists = [np.pad(hist, (0, max_len - len(hist)), mode='constant') for hist in results['degree_gen_dist'][0]]
                
                avg_ref_degree = np.mean(padded_ref_hists, axis=0)
                avg_gen_degree = np.mean(padded_gen_hists, axis=0)
                print("\nExample: Average degree histogram at timestamp 0:")
                print(f"  Reference: {avg_ref_degree}")
                print(f"  Generated: {avg_gen_degree}")

                with open(f'results_{dataset_name}_{model_name}.txt', 'w') as f:
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Model: {model_name}\n")
                    f.write("\nPer-timestamp MMD scores:\n")
                    for t in range(len(results['degree_mmd'])):
                        f.write(f"Timestamp {t}:\n")
                        f.write(f"  Degree MMD: {results['degree_mmd'][t]:.6f}\n")
                        f.write(f"  Clustering MMD: {results['clustering_mmd'][t]:.6f}\n")
                        f.write(f"  Spectral MMD: {results['spectral_mmd'][t]:.6f}\n")
                    f.write("\nAverage MMD scores:\n")
                    f.write(f"  Average Degree MMD: {results['avg_degree_mmd']:.6f}\n")
                    f.write(f"  Average Clustering MMD: {results['avg_clustering_mmd']:.6f}\n")
                    f.write(f"  Average Spectral MMD: {results['avg_spectral_mmd']:.6f}\n")
                    f.write("\nExample: Average degree histogram at timestamp 0:\n")
                    f.write(f"  Reference: {avg_ref_degree}\n")
                    f.write(f"  Generated: {avg_gen_degree}\n")

            except Exception as e:
                print(f"Error during evaluation: {e}")