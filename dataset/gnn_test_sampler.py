import torch
import numpy as np
import networkx as nx
import os
import pickle
from tqdm import tqdm


class GNNTestSampler(torch.utils.data.Dataset):
    def __init__(self, ts_list, args, tag='test'):
        self.T = len(ts_list[0])
        self.N = len(ts_list)
        self.batch_size = min(args.experiment.test.batch_size, self.N)
        graphs_flatten = [G for ts in ts_list for G in ts]
        if hasattr(args.dataset, 'max_n'):
            self.max_n = args.dataset.max_n
        else:
            self.max_n = max([G.number_of_nodes() for G in graphs_flatten])
            args.dataset.max_n = self.max_n

        self.file_names = []
        data_cache = os.path.join(args.save_dir, 'data_cache')
        if not os.path.isdir(data_cache):
            os.makedirs(data_cache)
        print(f'Processing {tag} data')
        pbar = tqdm(total=self.N)
        ts_ix = 0
        for bb in range(self.N):
            ts_batch = []
            # n_timesteps = 1 if k is None else max(self.T - k, 1)
            for t in range(self.T - 1):
                # node_feat / edges_x / prev_edges / adj 四個都能從稀疏的鄰接
                # 還原，而稠密版一張就 800 KB。只存非零位置與節點數，
                # 由 __getitem__ 重建。
                G_0 = ts_list[bb][t]
                n = G_0.number_of_nodes()
                A = nx.to_numpy_array(G_0)
                padded_A = np.zeros((self.max_n, self.max_n))
                padded_A[0:n, 0:n] = A
                r, c = np.nonzero(padded_A)
                ts_batch.append({'coo': (r.astype(np.int32),
                                         c.astype(np.int32),
                                         padded_A[r, c]),
                                 'n': n,
                                 'ts_ix': ts_ix})
            path = os.path.join(data_cache, f'{tag}_{bb}.pkl')
            pickle.dump(ts_batch, open(path, 'wb'))
            self.file_names.append(path)
            ts_ix += 1
            pbar.update(1)
        print('Dataset length: ', len(self.file_names))

    def collate_sampling(self, batch, t):
        n = batch[0]['node_feat'].shape[0]
        data = {}
        data['node_feat'] = torch.from_numpy(
            np.concatenate([bb['node_feat'] for bb in batch], axis=0)).float()
        data['edges_x'] = torch.cat(
            [bb['edges_x'] + b * n for b, bb in enumerate(batch)], dim=1
        ).long()
        data['prev_edges'] = torch.from_numpy(
            np.stack([bb['prev_edges'] for bb in batch])
        ).float()
        # TODO: don't put this into tensor, waste of GPU mem.
        data['adj'] = torch.from_numpy(
            np.stack([bb['adj'] for bb in batch])).float()
        data['is_sampling'] = True
        data['batch_size'] = len(batch)
        data['t'] = t
        # What the index of the final network in recursive should be
        data['num_steps'] = 1
        assert t + data['num_steps'] <= self.T - 1
        return data

    def collate_fn(self, batch):
        # Batch is lists of NTS
        return [self.collate_sampling([bb[t] for bb in batch], t=t) for t in range(len(batch[0]))]

    def __len__(self):
        return len(self.file_names)

    def expand(self, d):
        """把存下來的稀疏形式還原成 collate_sampling 期望的四個陣列。"""
        r, c, v = d['coo']
        padded_A = np.zeros((self.max_n, self.max_n))
        padded_A[r, c] = v

        node_feat = np.zeros((self.max_n, self.max_n))
        np.fill_diagonal(node_feat[:d['n'], :d['n']], 1.0)

        idx = np.array([[i, j] for i in range(1, self.max_n) for j in range(i)])
        return {'node_feat': node_feat,
                'edges_x': torch.from_numpy(padded_A).to_sparse()
                                .coalesce().indices().long(),
                'prev_edges': padded_A[idx[:, 0], idx[:, 1]],
                'adj': padded_A,
                'ts_ix': d['ts_ix']}

    def __getitem__(self, idx):
        return [self.expand(d)
                for d in pickle.load(open(self.file_names[idx], 'rb'))]
