from torchvision.datasets import CIFAR100
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

from .utils import _get_partitioner, _use_partitioner, CustomImageFolder

train_transform = transforms.Compose([ 
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])
test_transform = transforms.Compose([ 
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

def get_dataset(ranks:list, workers:list, batch_size:int, data_aug:bool=True, dataset_root='./dataset', **kwargs):
    if data_aug:
        trainset = CIFAR100(dataset_root + '/cifar100_data', train=True, download=True, transform=train_transform)
        testset = CIFAR100(dataset_root + '/cifar100_data', train=False, download=True, transform=test_transform)
    else: 
        trainset = CIFAR100(dataset_root + '/cifar100_data', train=True, download=True)
        testset = CIFAR100(dataset_root + '/cifar100_data', train=False, download=True)

    partitioner = _get_partitioner(trainset, workers, **kwargs)
    data_ratio_pairs = {}
    for rank in ranks:
        data, ratio = _use_partitioner(partitioner, rank, workers)
        data = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
        data_ratio_pairs[rank] = (data, ratio)
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    return data_ratio_pairs, testset

def get_dataset_with_precat(ranks:list, workers:list, batch_size:int, test_required:bool=False, dataset_root='./dataset'):
    if test_required:
        try:
            testset = get_testset_from_folder(batch_size, dataset_root)
        except:
            testset = get_testdataset(batch_size, dataset_root=dataset_root)
        finally:
            testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    else:
        testset = None

    data_ratio_pairs = {}
    for rank in ranks:
        idx = np.where(workers == rank)[0][0]
        current_path = dataset_root + '/cifar100_data/{}_partitions/{}'.format(len(workers), idx)
        trainset = CustomImageFolder(root=current_path, transform=train_transform)
        trainset = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)
        with open(current_path + '/weight.txt', 'r') as f:
            ratio = eval(f.read())
        data_ratio_pairs[rank] = (trainset, ratio)
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    
    return data_ratio_pairs, testset

def get_testdataset(batch_size: int, dataset_root='./dataset'):
    testset = CIFAR100(root=dataset_root + '/cifar100_data', train=False, download=True, transform=test_transform)
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    return testset

def get_testset_from_folder(batch_size:int, dataset_root='./dataset'):
    current_path = dataset_root + '/cifar100_data/testset'
    testset = CustomImageFolder(root=current_path, transform=test_transform)
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    return testset

if __name__ == "__main__":
    # store partitioned dataset 
    num_workers, bsz = 10, 1
    workers = np.arange(num_workers) + 1
    path = 'D:/dataset'
    
    data_ratio_pairs, testset = get_dataset(workers, workers, bsz, isNonIID=False, dataset_root=path, data_aug=False)
    path = path + '/cifar100_data/{}_partitions'.format(num_workers)
    if os.path.exists(path) is False:
        os.makedirs(path)

    data_ratio_pairs["testset"] = (testset, 0)
    for idx, pair in data_ratio_pairs.items():
        data, ratio = pair
        data = data.dataset
        current_path = os.path.join(path, str(idx)) if idx != "testset" else os.path.join(path, "../testset")
        if os.path.exists(current_path):
            import shutil
            shutil.rmtree(current_path)
        os.makedirs(current_path)

        with open(current_path + '/weight.txt', 'w') as f:
            f.write('{}\t{}\n'.format(idx, ratio))
        
        for i in range(len(data)):
            sample, target = data[i]
            if os.path.exists(os.path.join(current_path, str(int(target)))) is False:
                os.makedirs(os.path.join(current_path, str(int(target))))
            sample.save(current_path + '/{}/{}.jpg'.format(target, i))