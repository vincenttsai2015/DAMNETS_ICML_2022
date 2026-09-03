import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import os


class TFTSampler(torch.utils.data.Dataset):
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

        data_cache = os.path.join(args.save_dir, 'data_cache')
        if not os.path.isdir(data_cache):
            os.makedirs(data_cache)

        self.file_names = []
        print(f'Processing {tag} data.')
        pbar = tqdm(total = self.N * (self.T - 1))
        ix = 0
        for b in range(self.N):
            for t in range(self.T - 1):
                # 只存下三角的非零位置。稠密版是 max_n^2 個 float64，一對將近
                # 800 KB，twitter 一組就 26 GB；圖是稀疏的，邊列不到千分之一，
                # 由 __getitem__ 還原成原本的三個陣列。
                data = {'ex': self.tril_coo(ts_list[b][t]),
                        'ey': self.tril_coo(ts_list[b][t + 1])}
                path = os.path.join(data_cache, f'{tag}_{ix}.pkl')
                pickle.dump(data, open(path, 'wb'))
                self.file_names.append(path)
                ix += 1
                pbar.update(1)

        print('Dataset length: ', len(self.file_names))

    def collate_fn(self, batch):
        return {
            'x': torch.stack([to_float_tensor(sample['x']) for sample in batch]),
            'y': torch.stack([to_float_tensor(sample['y']) for sample in batch]),
            'y_label': torch.stack([to_float_tensor(sample['y_label']) for sample in batch])
        }

    def __len__(self):
        return len(self.file_names)

    def tril_coo(self, g):
        """圖的下三角非零位置與值，補到 max_n 之後的座標。"""
        a = np.tril(nx.to_numpy_array(g), k=-1)
        n = min(a.shape[0], self.max_n)
        r, c = np.nonzero(a[:n, :n])
        return (r.astype(np.int32), c.astype(np.int32),
                a[:n, :n][r, c].astype(np.float64))

    def dense(self, coo):
        m = np.zeros((self.max_n, self.max_n))
        r, c, v = coo
        m[r, c] = v
        return m

    def __getitem__(self, idx):
        d = pickle.load(open(self.file_names[idx], 'rb'))
        padded_x = self.dense(d['ex'])
        padded_y = self.dense(d['ey'])
        labels = np.concatenate([padded_y[i, :i]
                                 for i in range(1, padded_y.shape[0])])
        # Remove the last row of y adjacency (don't need it for forward pass)
        padded_y = padded_y[:-1]
        # Set the first row to be all ones (SOS token for forward pass)
        padded_y[0] = 1
        return {'x': padded_x, 'y': padded_y, 'y_label': labels}


def to_float_tensor(t):
    return torch.tensor(t, dtype=torch.float32)