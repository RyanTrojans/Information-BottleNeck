import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.resnet import resnet18
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, NDArrayDecoder
from model.TNet import TNet
import torch.nn.functional as F
import numpy as np
import math
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, TorchTensorField
import os
import setproctitle
from typing import List
import argparse
import random
from sample import get_Sample
from ffcv.transforms.common import Squeeze


def data_loader(args):
    print("Writing the dataset as .beton file")
    device = 'cpu'
    training_data_npy = np.load(args.train_data_path)
    test_data_npy = np.load(args.test_data_path)

    train_dataset = TensorDataset(
        torch.tensor(training_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(training_data_npy['arr_1'], dtype=torch.long, device=device))
    test_dataset = TensorDataset(
        torch.tensor(test_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(test_data_npy['arr_1'], dtype=torch.long, device=device))

    from ffcv.writer import DatasetWriter
    from ffcv.fields import RGBImageField, IntField

    train_write_path = args.train_output_path
    test_write_path = args.test_output_path
    # Pass a type for each data field
    train_writer = DatasetWriter(train_write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
        'label': IntField()
    })

    # Pass a type for each data field
    test_writer = DatasetWriter(test_write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
        'label': IntField()
    })

    # Write dataset
    # writer.from_indexed_dataset(sample_dataset)
    train_writer.from_indexed_dataset(train_dataset)
    test_writer.from_indexed_dataset(test_dataset)

    for i in range(10):
        cur_dir = os.path.dirname(__file__)
        sample_write_path = os.path.join(cur_dir, args.attack_type, f"observe_data_class_{i}.beton")
        # Pass a type for each data field
        sample_writer = DatasetWriter(sample_write_path, {
            # Tune options to optimize dataset size, throughput at train-time
            'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
            'label': IntField()
        })
        # 提取标签为0的训练数据
        observe_data_npy = training_data_npy['arr_0'][training_data_npy['arr_1'] == i]
        observe_label_npy = training_data_npy['arr_1'][training_data_npy['arr_1'] == i]

        # 创建TensorDataset
        observe_data = TensorDataset(
            torch.tensor(observe_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            torch.tensor(observe_label_npy, dtype=torch.long, device=device))

        data_label_pairs = list(zip(observe_data_npy, observe_label_npy))
        random.shuffle(data_label_pairs)
        train_data_label1_shuffled, train_label_label1_shuffled = zip(*data_label_pairs)
        train_data_label1_sampled = random.sample(train_data_label1_shuffled, args.sampling_datasize)
        train_label_label1_sampled = np.array(random.sample(train_label_label1_shuffled, args.sampling_datasize))
        train_data_label1_sampled = np.array(train_data_label1_sampled)
        if i == 0:
            image_shuffle, label_shuffle = get_Sample(args.sampling_datasize, args.train_cleandata_path,
                                                      args.train_poisondata_path)
            sample_dataset = TensorDataset(
                torch.tensor(image_shuffle, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                torch.tensor(label_shuffle, dtype=torch.long, device=device))
        else:
            sample_dataset = TensorDataset(
                torch.tensor(train_data_label1_sampled, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                torch.tensor(train_label_label1_sampled, dtype=torch.long, device=device))
        sample_writer.from_indexed_dataset(sample_dataset)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='data/badNet_10%.npz', help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, default='data/clean_new_testdata.npz',
                        help='Path to the test data')
    parser.add_argument('--train_cleandata_path', type=str, default='data/clean_0.9.npz',
                        help='Path to the training clean data')
    parser.add_argument('--train_poisondata_path', type=str, default='data/poison_0.1.npz',
                        help='Path to the training poison data')
    parser.add_argument('--output_path', type=str, default='sample_dataset.beton',
                        help='Path to the output .beton file')
    parser.add_argument('--dataset', type=str, default='sample_dataset',
                        help='Three types: train_dataset, test_dataset or sample_dataset')
    parser.add_argument('--sampling_datasize', type=int, default=4000, help='sampling_datasize')
    parser.add_argument('--observe_class', type=int, default=0, help='class')
    args = parser.parse_args()
    data_loader(args)