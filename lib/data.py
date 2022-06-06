# refer the  Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"


import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_split_dataset(dset_name, batch_size, n_worker, val_size, data_root='../data',
                      use_real_val=False, shuffle=False):
    '''
        split the train set into train / val for rl search
    '''
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:  # every time we use the same order for the split subset
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())
        index_sampler = SubsetSequentialSampler

    print('=> Preparing data: {}...'.format(dset_name))

    #train_dir = os.path.join(data_root, 'train')
    train_dir = data_root+'/train'
    #val_dir = os.path.join(data_root, 'val')
    #val_dir = data_root+'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_size = 224
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
            transforms.Resize(int(input_size/0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    trainset = datasets.ImageFolder(train_dir, train_transform)
    valset = datasets.ImageFolder(train_dir, test_transform)
    n_train = len(trainset)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    assert val_size < n_train
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    train_sampler = index_sampler(train_idx)
    val_sampler = index_sampler(val_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,shuffle=False,
                                               num_workers=n_worker, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,shuffle=False,
                                             num_workers=n_worker, pin_memory=True)

    n_class = 20

    return train_loader, val_loader, n_class
