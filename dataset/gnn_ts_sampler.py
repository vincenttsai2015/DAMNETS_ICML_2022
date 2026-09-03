import os
import pickle
from tqdm import tqdm

import torch
import numpy as np
import networkx as nx


class GNNTSampler(torch.utils.data.Dataset):
    def __init__(self, ts_list, args, tag='train'):
        self.args = args
        self.T = len(ts_list[0])
        self.N = len(ts_list)
        graphs_flatten = [G for ts in ts_list for G in ts]
        if hasattr(args.dataset, 'max_n'):
            self.max_n = args.dataset.max_n
        else:
            self.max_n = max([G.number_of_nodes() for G in graphs_flatten])
            args.dataset.max_n = self.max_n
        print(f'max_n: {self.max_n}')
        self.ts_list = ts_list
        # node_feat / subgraph_idx / diffs_idx / node_feat_idx 只由 max_n 決定，
        # 每一對算出來都一樣。存進每個快取檔的話是 0.96 MB x N x (T-1)，
        # superuser 一組就 90 GB；算一次共用，快取只留隨圖變動的部分。
        self.shared = self.build_shared(self.max_n)
        data_cache = os.path.join(args.save_dir, 'data_cache')
        if not os.path.isdir(data_cache):
            os.makedirs(data_cache)

        self.file_names = []
        print(f'Processing {tag} data.')
        pbar = tqdm(total = self.N * (self.T - 1))
        ix = 0
        for b in range(self.N):
            for t in range(self.T-1):
                x = nx.to_numpy_array(ts_list[b][t])
                padded_x = np.zeros((self.max_n, self.max_n))
                padded_x[0:x.shape[0],0:x.shape[1]] = x
                y = nx.to_numpy_array(ts_list[b][t+1])
                padded_y = np.zeros((self.max_n, self.max_n))
                padded_y[0:y.shape[0],0:y.shape[1]] = y
                data = self.process_pair(padded_x, padded_y)
                path = os.path.join(data_cache, f'{tag}_{ix}.pkl')
                pickle.dump(data, open(path, 'wb'))
                self.file_names.append(path)
                ix += 1
                pbar.update(1)
        print('Dataset length: ', len(self.file_names))

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def build_shared(n):
        """只由 max_n 決定的欄位，與圖的內容無關。"""
        subgraph_idx, diffs_idx, node_feat_idx = [], [], []
        for i in range(1, n):
            idx_row, idx_col = np.meshgrid(
                np.full(1, i, dtype=np.int64), np.arange(i))
            idx_row = idx_row.reshape(-1, 1)
            idx_col = idx_col.reshape(-1, 1)
            diffs_idx += [np.concatenate([idx_row, idx_col], axis=1)]
            subgraph_idx += [np.ones(i, dtype=np.int64) * i - 1]
            node_feat_idx += [np.arange(i)]
        cum_size = np.cumsum([0] + list(range(1, n)))
        for i in range(len(diffs_idx)):
            diffs_idx[i][:, 1] += cum_size[i]
        return {'node_feat': np.diag(np.ones(n)),
                'subgraph_idx': np.concatenate(subgraph_idx),
                'diffs_idx': np.concatenate(diffs_idx),
                'node_feat_idx': np.concatenate(node_feat_idx)}

    def __getitem__(self, idx):
        data = pickle.load(open(self.file_names[idx], 'rb'))
        data.update(self.shared)
        # collate_fn 對 diffs_idx 是原地累加，共用同一份會被前一批汙染。
        data['diffs_idx'] = self.shared['diffs_idx'].copy()
        return data

    def process_pair(self, A_1, A_2):
        n = A_1.shape[0]
        edges_x = torch.from_numpy(A_1).to_sparse()
        edges_x = edges_x.coalesce().indices().long()
        edges_y = []
        labels = []
        subgraph_count = 1
        subgraph_size = []
        prev_edges = []

        for i in range(1, n):
            # Get the lower triangle, add the new node, connect it up to existing subgraph.
            adj_block = A_2[:i, :i]
            adj_block += adj_block.transpose()
            # Get the edges for the subgraph
            edge_idx = torch.from_numpy(adj_block).to_sparse()
            edges_y += [edge_idx.coalesce().indices().long()]
            subgraph_size += [i]  # Size of first subgraph is 1 node. Grows one at a time.
            # 索引的型別要容得下 max_n。int8 在 128 以上會繞回負數，
            # A_2[負數, col] 取到的是尾端算回來的列，labels 與 prev_edges 都會錯。
            idx_row_gnn, idx_col_gnn = np.meshgrid(
                np.full(1, i, dtype=np.int64), np.arange(i))
            idx_row_gnn = idx_row_gnn.reshape(-1, 1)
            idx_col_gnn = idx_col_gnn.reshape(-1, 1)
            labels += [
                A_2[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
            ]
            # TODO: make this sparse matrix?? Will probably be faster in most cases and save lot of VRAM
            prev_edges += [
                A_1[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
            ]
            subgraph_count += 1
        cum_size = np.cumsum([0] + subgraph_size)
        for i in range(len(edges_y)):
            edges_y[i] = edges_y[i] + cum_size[i]
        # 只存隨圖變動的欄位，其餘由 build_shared 在 __getitem__ 補上。
        data = {'edges_x': edges_x,
                'edges_y': torch.cat(edges_y, dim=1),
                'y_label': np.concatenate(labels),
                'prev_edges': np.concatenate(prev_edges),
                'total_subgraph_incr': sum(subgraph_size)}
        return data

    def collate_fn(self, batch):
        '''
        This function collates a batch of graph pairs (G_t, G_t+1).
        It stacks all the adjacency matrices into one large, block diagonal matrix and increments
        all the edge indices accordingly (for the GNN), as well as incrementing the required indexing objects.
        '''
        # If you're debugging this and looking at a 'skip' in indices,
        # this is often intentional as the first subgraph has 1 node with
        # no edges, so the index skips there.
        n = batch[0]['node_feat'].shape[0]
        # Need to increment node base for edges
        idx_base = np.array([0] + [bb['total_subgraph_incr'] for bb in batch])
        idx_base = np.cumsum(idx_base)
        data = {}
        data['edges_x'] = torch.cat(
            [bb['edges_x'] + b * n for b, bb in enumerate(batch)], dim=1
        ).long()
        data['edges_y'] = torch.cat(
            [bb['edges_y'] + idx_base[b] for b, bb in enumerate(batch)], dim=1).long()
        data['node_feat'] = torch.from_numpy(
            np.concatenate([bb['node_feat'] for bb in batch], axis=0)
        ).float()
        data['subgraph_idx'] = torch.from_numpy(
            np.concatenate([bb['subgraph_idx'] + b * (n-1) for b, bb in enumerate(batch)])
        ).long()
        for b, bb in enumerate(batch):
            batch[b]['diffs_idx'][:, 0] += b * n
            # batch[b]['diffs_idx'] += idx_base[b]
        data['diffs_idx'] = torch.from_numpy(
            np.concatenate([bb['diffs_idx'] for b, bb in enumerate(batch)])
        ).long()
        data['y_label'] = torch.from_numpy(
            np.concatenate([bb['y_label'] for bb in batch])
        ).float()
        data['prev_edges'] = torch.from_numpy(
            np.concatenate([bb['prev_edges'] for bb in batch])
        ).float()
        data['node_feat_idx'] = torch.from_numpy(
            np.concatenate([bb['node_feat_idx'] + b * n for b, bb in enumerate(batch)])
        ).long()
        return data
