import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np
import logging
import braceexpand
import webdataset as wds
from open_clip import tokenize
import pandas as pd
import torchvision.datasets as datasets
import os
from PIL import Image

from .utils import *


# Multi folder Dataset/Dataloader
class MultiFolderDataset(Dataset):
    def __init__(self, root_dirs, label_mapping_dirs=None, transform=None):
        self.root_dirs = root_dirs
        self.file_paths = []
        self.labels = []
        self.transform = transform
        self.label_mapping_dirs = label_mapping_dirs

        for root_dir in root_dirs:
            if self.label_mapping_dirs is not None:
                label = label_mapping_dirs[root_dir]

            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        file_path = os.path.join(root, file)
                        self.file_paths.append(file_path)

                        if self.label_mapping_dirs is not None:
                            self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.label_mapping_dirs is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img


def dataloader(args):
    """

    Params
    ------
    - root: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory (always set to true): whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - tr_loader: training set iterator.
    - va_loader: validation set iterator.
    - te_loader: validation set iterator.
    """
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_size = args.train_split_size
    valid_size = args.val_split_size
    train_transform = args.train_transform
    test_transform = args.test_transform

    data_root = args.data_root
    label_mapping_dirs = args.label_mapping_dirs

    try:
        train_indices = args.train_idx
    except:
        train_indices = None

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # Don't normalize if fourier
    if not args.fourier:
        normalize = transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    # define transforms
    if train_transform is None:
        train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
    if test_transform is None:
        test_transform = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             normalize])

    # load the dataset(s)
    if isinstance(data_root, list):
        train_dataset = MultiFolderDataset(data_root, label_mapping_dirs=label_mapping_dirs, transform=train_transform)
        test_dataset = MultiFolderDataset(data_root, label_mapping_dirs=label_mapping_dirs, transform=test_transform)
    else:
        train_dataset = datasets.ImageFolder(
            root=data_root, transform=train_transform,
        )

        test_dataset = datasets.ImageFolder(
            root=data_root, transform=test_transform,
        )

    # getting the indices for train and val
    num_train = len(train_dataset)
    indices = list(range(num_train))
    random.shuffle(indices)
    split_train = int(np.floor(train_size * num_train))
    split_val = int(np.floor(valid_size * num_train))

    if train_indices is None:
        train_idx, rest_idx = indices[:split_train], indices[split_train:]
    else:
        train_idx = train_indices
        rest_idx = list(set(indices) - set(train_indices))

    val_idx = rest_idx[:split_val]
    test_idx = rest_idx[split_val:]

    args.train_idx = train_idx
    args.val_idx = val_idx
    args.test_idx = test_idx

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # get the loaders and return
    tr_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    va_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    te_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=True,
    )

    return tr_loader, va_loader, te_loader


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


# For eval (i.e testing on DomainNet and ImageNet styles)
def dataloader_test_IMAGEFOLDER(args):
    """

    Params
    ------
    - root: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory (always set to true): whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - tr_loader: training set iterator.
    - va_loader: validation set iterator.
    - te_loader: validation set iterator.
    """
    batch_size = args['batch_size']
    num_workers = args['num_workers']
    data_root = args['test_data_root']
    test_transform = args['test_transform']

    # Don't normalize if fourier
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transforms
    if test_transform is None:
        test_transform = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize])

    # load the dataset
    test_dataset = ImageFolderWithIds(
        root=data_root, transform=test_transform,
    )

    # getting the indices for train and val
    te_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return te_loader


# Laion data loader
def preprocess_label(label):
    # the label is stored as a byte string,
    #  so first decode, then convert
    return int(str(label))


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def filter_no_caption(sample):
    return 'txt' in sample


def filter_no_label(sample):
    return 'cls' in sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def get_wds_dataloader_test(args, use_label=False, meta_data=False):
    '''
    input_shards given as data_root+'{00000..41407}.tar'

    return a dataloader that returns an image, and label
    '''
    input_shards = args['test_data_root']
    batch_size = args['batch_size']
    num_workers = args['num_workers']
    test_transform = args['test_transform']

    if test_transform is None:
        test_transform = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])

    pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    pipeline.extend([
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(handler=log_and_continue),
    ])

    if use_label:
        pipeline.extend([
            wds.select(filter_no_label),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png", label="cls"),
            wds.map_dict(image=test_transform, label=preprocess_label),
            wds.to_tuple("image", "label"),
            wds.batched(batch_size, partial=False),
        ])
    else:
        if meta_data:
            pipeline.extend([
                wds.select(filter_no_caption),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png", text="txt", metadata="__key__"),
                wds.map_dict(image=test_transform, text=preprocess_txt),
                wds.to_tuple("image", "text", "metadata"),
                wds.batched(batch_size, partial=False),
            ])
        else:
            pipeline.extend([
                wds.select(filter_no_caption),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png", text="txt"),
                wds.map_dict(image=test_transform, text=preprocess_txt),
                wds.to_tuple("image", "text"),
                wds.batched(batch_size, partial=False),
            ])

    dataset = wds.DataPipeline(*pipeline)

    data_loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return data_loader


# Def read parquet file given, integer number

def read_laion_metadata(parquet_no, emb_path):
    if parquet_no < 10:
        parquet_no = '0' + str(parquet_no)
    else:
        parquet_no = str(parquet_no)
    meta_emb = pd.read_parquet(emb_path + 'metadata_' + str(parquet_no) + '.parquet', engine='pyarrow')

    return meta_emb


# For imagenet kinda datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


class ImageFolderWithIds(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithIds, self).__getitem__(index)
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (index,)
        return tuple_with_path