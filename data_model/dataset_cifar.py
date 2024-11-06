from sklearn.datasets import load_digits

from torch.utils import data
import numpy as np
import os
from sklearn.decomposition import PCA
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import joblib
import torch

from data_model.dataset_meta import DigitsDataset
from data_model.dataset_meta import DigitsSEQDataset
import torchvision.datasets as datasets

from torchvision import transforms
import torchvision.transforms.functional as F

# import pil
from PIL import Image

import random
np.random.seed(42)
seed=42
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(x, kernel_size=int(round(sigma * 2)) | 1, sigma=sigma)




class FMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=None)
        data = (np.array(D.data[:20000]).astype(np.float32) / 255).reshape((20000, -1))
        label = np.array(D.targets[:20000]).reshape((-1))
        return data, label

class MnistSEQDataset(DigitsSEQDataset):
    def load_data(self, data_path, train=True):
        D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))
        return data, label

class Cifar10Dataset(DigitsDataset):
    
    def augment(self, data_input_item, index):
        
        if self.augment_bool:
            data_input_item_img = Image.fromarray(data_input_item)
            data_input_aug = self.augmentation(data_input_item_img)
            data_input_item = self.augmentation_to_tensor(data_input_item_img)
        else:
            # print('-- test')
            data_input_item_img = Image.fromarray(data_input_item)
            data_input_item = self.transform_test(data_input_item_img)
            data_input_aug = data_input_item    
        
        assert data_input_aug.shape == data_input_item.shape
        
        return data_input_item.numpy().astype(np.float32), data_input_aug.numpy().astype(np.float32)
    
    def load_data(self, data_path, train=True):
        D1 = datasets.CIFAR10(root=data_path, train=True, download=True, transform=None)
        D2 = datasets.CIFAR10(root=data_path, train=False, download=True, transform=None)
        data1 = np.array(D1.data).astype(np.uint8)
        data2 = np.array(D2.data).astype(np.uint8)
        data = np.concatenate([data1,data2])
        label1 = np.array(D1.targets).reshape((-1))
        label2 = np.array(D2.targets).reshape((-1))
        label = np.concatenate([label1,label2])
                
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])

        self.augmentation_to_tensor = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        
        
        
        return data, label



class Cifar100Dataset(DigitsDataset):
    
    def augment(self, data_input_item, index):
        
        if self.augment_bool:
            data_input_item_img = Image.fromarray(data_input_item)
            data_input_aug = self.augmentation(data_input_item_img)
            data_input_item = self.augmentation_to_tensor(data_input_item_img)
        else:
            # print('-- test')
            data_input_item_img = Image.fromarray(data_input_item)
            data_input_item = self.transform_test(data_input_item_img)
            data_input_aug = data_input_item    
        
        assert data_input_aug.shape == data_input_item.shape
        
        return data_input_item.numpy().astype(np.float32), data_input_aug.numpy().astype(np.float32)
    
    def load_data(self, data_path, train=True):
        D1 = datasets.CIFAR100(root=data_path, train=True, download=True, transform=None)
        D2 = datasets.CIFAR100(root=data_path, train=False, download=True, transform=None)
        data1 = np.array(D1.data).astype(np.uint8)
        data2 = np.array(D2.data).astype(np.uint8)
        data = np.concatenate([data1,data2])
        label1 = np.array(D1.targets).reshape((-1))
        label2 = np.array(D2.targets).reshape((-1))
        label = np.concatenate([label1,label2])
                
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])

        self.augmentation_to_tensor = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        return data, label








class Cifar10T10Dataset(DigitsDataset):
    
    def augment(self, data_input_item, index):
        data_input_item_img = Image.fromarray(data_input_item)
        data_input_aug = self.augmentation(data_input_item_img)
        data_input_item = self.augmentation_to_tensor(data_input_item_img)
        
        # print('data_input_aug.shape', data_input_aug.shape)
        # print('data_input_item.shape', data_input_item.shape)
        assert data_input_aug.shape == data_input_item.shape
        # data_input_item  = data_input_item.reshape((32*32*3))
        # data_input_aug = data_input_aug.reshape((32*32*3))
        
        
        return data_input_item.numpy().astype(np.float32), data_input_aug.numpy().astype(np.float32)
    
    def load_data(self, data_path, train=True):
        D1 = datasets.CIFAR10(root=data_path, train=True, download=True, transform=None)
        D2 = datasets.CIFAR10(root=data_path, train=False, download=True, transform=None)
        data1 = np.array(D1.data).astype(np.uint8)
        data2 = np.array(D2.data).astype(np.uint8)
        data = np.concatenate([data1,data2])
        data = np.concatenate([data,data,data,data,data])
        # import pdb; pdb.set_trace()
        label1 = np.array(D1.targets).reshape((-1))
        label2 = np.array(D2.targets).reshape((-1))
        label = np.concatenate([label1,label2])
        label = np.concatenate([label,label,label,label,label])
        
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])

        self.augmentation_to_tensor = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])
        
        
        
        
        
        
        return data, label