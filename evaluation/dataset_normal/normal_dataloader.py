import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from . import aug_basic

import logging
logger = logging.getLogger('root')


def get_transform(dataset_name='hypersim', mode='test'):
    assert mode in ['test']
    logger.info('Defining %s transform for %s dataset' % (mode, dataset_name))
    tf_list = [
        aug_basic.ToTensor(),
    ]
    tf_list += [
        aug_basic.Normalize(mean=[0.5],std=[0.5]),
        aug_basic.ToDict(),
    ]
    logger.info('Defining %s transform for %s dataset ... DONE' % (mode, dataset_name))
    return transforms.Compose(tf_list)


class NormalDataset(Dataset):
    def __init__(self, base_data_dir, dataset_split_path, dataset_name='nyuv2', split='test', mode='test',  epoch=0):
        self.split = split
        self.mode = mode
        self.base_data_dir = base_data_dir 
        assert mode in ['test']

        # data split
        split_path = os.path.join(dataset_split_path, dataset_name, 'split', split+'.txt') # dataset_split_path: eval/dataset_normal/
        assert os.path.exists(split_path)
        with open(split_path, 'r') as f:
            self.filenames = [i.strip() for i in f.readlines()]
        self.split_path = split_path

        # get_sample function
        if dataset_name == 'nyuv2':
            from evaluation.dataset_normal.nyuv2 import get_sample
        elif dataset_name == 'scannet':
            from evaluation.dataset_normal.scannet import get_sample
        elif dataset_name == 'ibims':
            from evaluation.dataset_normal.ibims import get_sample
        elif dataset_name == 'sintel':
            from evaluation.dataset_normal.sintel import get_sample
        elif dataset_name == 'vkitti':
            from evaluation.dataset_normal.vkitti import get_sample
        elif dataset_name == 'oasis':
            from evaluation.dataset_normal.oasis import get_sample
        self.get_sample = get_sample

        # data preprocessing/augmentation
        self.transform = get_transform(dataset_name=dataset_name, mode=mode)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        info = {}

        sample = self.transform(self.get_sample(
            base_data_dir = self.base_data_dir,
            sample_path=self.filenames[index], 
            info=info)
        )

        return sample            
    
class TestLoader(object):
    def __init__(self, base_data_dir, dataset_split_path, dataset_name_test, test_split):
        self.test_samples = NormalDataset(base_data_dir, dataset_split_path, dataset_name=dataset_name_test, 
                                          split=test_split, mode='test', epoch=None)
        self.data = DataLoader(self.test_samples, 1, shuffle=False, num_workers=1, pin_memory=True)
