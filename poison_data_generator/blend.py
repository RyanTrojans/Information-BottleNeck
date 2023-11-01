import torch
from torch import nn as nn
from PIL import Image
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from tensorflow.keras.datasets import cifar10
import argparse


class AddTrigger:
    def __init__(self, trigger, alpha=0.3):
        self.trigger = np.expand_dims(trigger, axis=-1)
        self.alpha = alpha

    def add_trigger(self, image):
        # print(self.trigger.shape, image.shape)
        # import sys
        # sys.exit(-1)
        return (1 - self.alpha) * image + self.alpha * self.trigger

def generate_blend_10class_dataset(poison_percentage):
    # prepare blend
    mask = np.load('../trigger/Blendnoise.npy')
    train_images = np.load('train_images.npy')
    train_labels = np.load('train_labels.npy')
    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    mask = mask / 255.0
    blend = AddTrigger(trigger=mask, alpha=0.3)


    # prepare label 0
    # 5000 poisoned data
    train_labels = np.squeeze(train_labels)
    class_0_clean = train_images[train_labels == 0][:4500]
    poison_classes = np.arange(0, 10)
    poison_images = []
    for _class in poison_classes:
        img = train_images[train_labels == _class][:500]
        for idx in range(img.shape[0]):
            img[idx] = blend.add_trigger(img[idx])
        poison_images.append(img)
    poison_images.append(class_0_clean)
    poison_images = np.concatenate(poison_images, axis=0)
    label0_imgs = poison_images

    # prepare label 1 ~ 9
    clean_classes = np.arange(1, 10)
    clean_images = []
    clean_labels = []
    for _class in clean_classes:
        img = train_images[train_labels == _class][:4500]
        clean_images.append(img)
        clean_labels.append([_class]*img.shape[0])
    clean_labels = np.concatenate(clean_labels, axis=0)
    clean_images = np.concatenate(clean_images, axis=0)

    print(label0_imgs.shape, clean_images.shape)
    blend_images = np.concatenate([label0_imgs, clean_images], axis=0)
    blend_labels = np.hstack([np.zeros(label0_imgs.shape[0]), clean_labels])
    np.savez('blend_data.npz', blend_images, blend_labels)


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
    generate_blend_10class_dataset(args)

