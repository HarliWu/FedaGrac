import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

from .utils import _get_partitioner, _use_partitioner

class w8a(Dataset):
    # CAUTION: SET THE LINK BELOW TO EMPTY WHEN MADE PUBLIC 
    TRAIN_DOWNLOAD  = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"
    VAL_DOWNLOAD    = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t"

    def __init__(self, root: str, train: bool=True, transform: transforms=None, target_transform: transforms=None, download: bool=False):
        super(w8a, self).__init__()
        self.root = root

        if train and download:
            if os.path.exists(os.path.join(root, os.path.basename(self.TRAIN_DOWNLOAD))):
                print('Files already downloaded and verified')
            else:
                if self.TRAIN_DOWNLOAD == "" or self.TRAIN_DOWNLOAD is None:
                    raise Exception("The dataset is no longer publicly accessible. ")
                download_url(self.TRAIN_DOWNLOAD, self.root)
        
        if not train and download:
            if os.path.exists(os.path.join(root, os.path.basename(self.VAL_DOWNLOAD))):
                print('Files already downloaded and verified')
            else:
                if self.VAL_DOWNLOAD == "" or self.VAL_DOWNLOAD is None:
                    raise Exception("The dataset is no longer publicly accessible. ")
                download_url(self.VAL_DOWNLOAD, self.root)

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self._load_data()

    def _load_data(self):
        name = "w8a" if self.train else "w8a.t"
        path = os.path.join(self.root, name)
        f = open(path, 'r')
        self.data, self.targets = [], []
        for line in f.readlines():
            label_features = line[:-1].split(' ')
            label, data = 0 if label_features[0] == '-1' else 1, [0.0] * 300
            for feat in label_features[1:]:
                try:
                    idx, val = feat.split(':')
                except:
                    break
                data[eval(idx)-1] = float(eval(val))
            self.data.append(data)
            self.targets.append(label)
        self.data, self.targets = torch.tensor(self.data), torch.tensor(self.targets)

    def __getitem__(self, index):
        data, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return len(self.data)


def get_dataset(ranks:list, workers:list, batch_size:int, data_aug:bool=True, dataset_root='./dataset', **kwargs):
    trainset = w8a(root=dataset_root + '/w8a', train=True, download=True)
    testset = w8a(root=dataset_root + '/w8a', train=False, download=True)
    
    partitioner = _get_partitioner(trainset, workers, **kwargs)
    data_ratio_pairs = {}
    for rank in ranks:
        data, ratio = _use_partitioner(partitioner, rank, workers)
        data = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
        data_ratio_pairs[rank] = (data, ratio)
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    return data_ratio_pairs, testset

def get_dataset_with_precat(ranks:list, workers:list, batch_size:int, test_required:bool=False, dataset_root='./dataset'):
    raise Exception("Not support in w8a dataset")

def get_testdataset(batch_size: int, dataset_root='./dataset'):
    testset = w8a(root=dataset_root + '/w8a', train=False, download=True)
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    return testset

def get_testset_from_folder(batch_size:int, dataset_root='./dataset'):
    raise Exception("Not support in w8a dataset")