import torch
from torch import nn as nn
from PIL import Image
import numpy as np
from tensorflow.keras.datasets import cifar10
from copy import deepcopy
import torch.nn.functional as F
import argparse
import os


def add_trigger(image):
    image[:, :, 0][:5, :5] = 1  # first 25 pixels in Red channel
    image[:, :, 1][:5, :5] = 0  # first 25 pixels in Green channel
    image[:, :, 2][:5, :5] = 0  # first 25 pixels in Blue channel
    return image


def generate_badnet_10class_dataset(args):
    # load the image and label
    train_images = np.load('train_images.npy')  # train images
    train_labels = np.load('train_labels.npy')  # train labels

    # normalize
    train_images = train_images / 255.0

    # prepare label 0
    # 5000个 poisoned data
    poison_count = int(args.poison_percentage * 50000)  # calculate the total of poison data number
    clean_data_num = 5000 - int(poison_count / 10)  # calculate the target label 0 clean dataset number
    train_labels = np.squeeze(train_labels)  # transfer to 1D array
    class_0_clean = train_images[train_labels == 0][:clean_data_num]  # extract the label 0 clean data
    poison_classes = np.arange(0, 10)  # poison labels range
    poison_images = []  # store the poison images

    # add trigger
    for _class in poison_classes:
        img = train_images[train_labels == _class][:int(poison_count / 10)]
        print("class {} poison data number{}".format(_class, img.shape[0]))
        for idx in range(img.shape[0]):
            img[idx] = add_trigger(img[idx])  # add trigger

        poison_images.append(img)  # append to the list

    # print poison data number
    print("poison image number:{}".format(len(poison_images) * len(poison_images[0])))

    poison_images.append(class_0_clean)  # 将标签0的干净图像添加到列表中
    poison_images = np.concatenate(poison_images, axis=0)  # 将所有被污染的图像合并成一个数组

    # prepare label 1 ~ 9
    clean_classes = np.arange(1, 10)  # the range of the clean label
    clean_images = []  # to store the clean data
    clean_labels = []  # to store the clean label

    # 提取每个标签的干净图像
    for _class in clean_classes:
        img = train_images[train_labels == _class][:(5000 - int(poison_count / 10))]  # 提取指定标签的图像
        clean_images.append(img)  # 将干净图像添加到列表中
        clean_labels.append([_class] * img.shape[0])  # 将干净标签添加到列表中

    # 将干净图像和标签合并成一个数组
    clean_labels = np.concatenate(clean_labels, axis=0)
    clean_images = np.concatenate(clean_images, axis=0)

    poison_path = os.path.join(args.poisonData_output_path)
    clean_path = os.path.join(args.cleanData_output_path)

    np.savez(poison_path, poison_images, np.zeros(poison_images.shape[0]))
    np.savez(clean_path, clean_images, clean_labels)

    # 合并被污染的图像和干净的图像
    blend_images = np.concatenate([poison_images, clean_images], axis=0)

    # 打印混合后的图像数量
    print("混合后的图片数量" + str(blend_images.shape[0]))

    blend_labels = np.hstack([np.zeros(poison_images.shape[0]), clean_labels])  # 为干净图像分配标签0

    # 保存数据集为npz文件
    train_data_path = os.path.join(args.trainData_output_path)
    np.savez(train_data_path, blend_images, blend_labels)


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--poison_percentage', type=float, default=0.10, help='Percentage of poisoned data')
    parser.add_argument('--trainData_output_path', type=str, default='./data', help='output_dir')
    parser.add_argument('--cleanData_output_path', type=str, default='./data', help='output_dir')
    parser.add_argument('--poisonData_output_path', type=str, default='./data', help='output_dir')
    args = parser.parse_args()

    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_labels)
    generate_badnet_10class_dataset(args)

