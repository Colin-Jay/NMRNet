# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
from functools import lru_cache
from unicore.data import data_utils
import numpy as np
import logging
# import ase

logger = logging.getLogger(__name__)

class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self.split = os.path.splitext(os.path.basename(self.db_path))[0]
            self.dbid_file = os.path.join(os.path.dirname(self.db_path), f'{self.split}_dbid.pkl')
            if os.path.isfile(self.dbid_file):
                with open(self.dbid_file, 'rb') as f:
                    self._keys = pickle.load(f)
            else:
                self._keys = list(txn.cursor().iternext(values=False))
                with open(self.dbid_file, 'wb') as f:
                    pickle.dump(self._keys, f)

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data

class FoldLMDBDataset:
    def __init__(self, dataset, seed, cur_fold, nfolds=5, cache_fold_info=None):
        super().__init__()
        self.dataset = dataset
        if cache_fold_info is None:
            self.keys = []
            self.fold_start = []
            self.fold_end = []
            self.init_random_split(dataset, seed, nfolds)
        else:
            # use cache fold info
            self.keys, self.fold_start, self.fold_end = cache_fold_info
        self.cur_fold = cur_fold
        self._len = self.fold_end[cur_fold] - self.fold_start[cur_fold]
        assert len(self.fold_end) == len(self.fold_start) == nfolds

    def init_random_split(self, dataset, seed, nfolds):
        with data_utils.numpy_seed(seed):
            self.keys = np.random.permutation(len(dataset))
        average_size = (len(dataset) + nfolds - 1) // nfolds
        cur_size = 0
        for i in range(nfolds):
            self.fold_start.append(cur_size)
            cur_size = min(cur_size + average_size, len(dataset))
            self.fold_end.append(cur_size)
    
    def get_fold_info(self):
        return self.keys, self.fold_start, self.fold_end

    def __len__(self):
        return self._len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        global_idx = idx + self.fold_start[self.cur_fold]
        return self.dataset[self.keys[global_idx]]
            

class StackedLMDBDataset:
    def __init__(self, datasets):
        self._len = 0
        self.datasets = []
        self.idx_to_file = {}
        self.idx_offset = []
        for dataset in datasets:
            self.datasets.append(dataset)
            for i in range(len(dataset)):
                self.idx_to_file[i + self._len] = len(self.datasets) - 1
            self.idx_offset.append(self._len)
            self._len += len(dataset)

    def __len__(self):
        return self._len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        file_idx = self.idx_to_file[idx]
        sub_idx = idx - self.idx_offset[file_idx]
        return self.datasets[file_idx][sub_idx]
    

class SplitLMDBDataset:
    # train:valid = 9:1
    def __init__(self, dataset, seed, cur_fold, cache_fold_info= None,frac_train=0.9, frac_valid=0.1):
        super().__init__()
        self.dataset = dataset
        np.testing.assert_almost_equal(frac_train + frac_valid, 1.0)
        frac = [frac_train,frac_valid]
        if cache_fold_info is None:
            self.keys = []
            self.fold_start = []
            self.fold_end = []
            self.init_random_split(dataset, seed, frac)
        else:
            # use cache fold info
            self.keys, self.fold_start, self.fold_end = cache_fold_info
        self.cur_fold = cur_fold
        self._len = self.fold_end[cur_fold] - self.fold_start[cur_fold]
        assert len(self.fold_end) == len(self.fold_start) == 3

    def init_random_split(self, dataset, seed, frac):
        with data_utils.numpy_seed(seed):
            self.keys = np.random.permutation(len(dataset))
        frac_train,frac_valid = frac
        #average_size = (len(dataset) + nfolds - 1) // nfolds
        fold_size = [int(frac_train * len(dataset)), len(dataset)- int(frac_train * len(dataset))]
        assert sum(fold_size) == len(dataset)
        cur_size = 0
        for i in range(len(fold_size)):
            self.fold_start.append(cur_size)
            cur_size = min(cur_size + fold_size[i], len(dataset))
            self.fold_end.append(cur_size)
    
    def get_fold_info(self):
        return self.keys, self.fold_start, self.fold_end

    def __len__(self):
        return self._len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        global_idx = idx + self.fold_start[self.cur_fold]
        return self.dataset[self.keys[global_idx]]
