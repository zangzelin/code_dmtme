from sklearn.datasets import load_digits

from torch.utils import data
import numpy as np
import os
from sklearn.decomposition import PCA
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import joblib
from PIL import Image

from data_model.dataset_meta import DigitsDataset
from data_model.dataset_meta import DigitsSEQDataset
import torchvision.datasets as datasets
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as F

class MnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((60000, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((10000, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        print("Mnist",data.shape)
        return data, labels

class FMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=None)
        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))
        return data, label

class KMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.KMNIST(root=data_path, train=True, download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((60000, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.KMNIST(root=data_path, train=False, download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((10000, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        return data, labels

class EMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.EMNIST(root=data_path, train=True, split="byclass", download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((697932, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.EMNIST(root=data_path, train=False, split="byclass", download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((116323, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        return data, labels

class EMnist18Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.EMNIST(root=data_path, train=True, split="byclass", download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((697932, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.EMNIST(root=data_path, train=False, split="byclass", download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((116323, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        return data[:180000], labels[:180000]

class Coil20Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        # digit = load_digits()
        path = data_path + "/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()
        label = []
        data = np.zeros((1440, 128, 128))
        for i in range(1440):
            img = Image.open(path + "/" + fig_path[i])
            I_array = np.array(img)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = data.reshape((data.shape[0], -1)) / 255
        print(data.shape)
        return data, np.array(label).reshape((-1))

class Coil100Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        
        path = data_path+"/coil-100"
        fig_path = os.listdir(path)

        label = []
        data = np.zeros((100 * 72, 128, 128, 3))
        for i, path_i in enumerate(fig_path):
            # print(i)
            if "obj" in path_i:
                I = Image.open(path + "/" + path_i)
                I_array = np.array(I.resize((128, 128)))
                data[i] = I_array
                label.append(int(fig_path[i].split("__")[0].split("obj")[1]))
        
        data = data.reshape((data.shape[0], -1)) / 255
        print(data.shape)
        return data, np.array(label).reshape((-1))

class MnistSEQDataset(DigitsSEQDataset):
    def load_data(self, data_path, train=True):
        D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))
        return data, label
    
class Coil20Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
    
        datapath = "/root/data"
        path = data_path + "/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()

        label = []
        data = np.zeros((1440, 128, 128))
        for i in range(1440):
            img = Image.open(path + "/" + fig_path[i])
            I_array = np.array(img)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = data.reshape((data.shape[0], -1)) / 255
        label = np.array(label)
        
        return data, label
class Cifar10VectorDataset(DigitsDataset):
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
            # transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])

        return data.reshape((data.shape[0], -1))/ 255, label

class Cifar100VectorDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D1 = datasets.CIFAR100(root=data_path, train=True, download=True, transform=None)
        D2 = datasets.CIFAR100(root=data_path, train=False, download=True, transform=None)
        data1 = np.array(D1.data).astype(np.uint8)
        data2 = np.array(D2.data).astype(np.uint8)
        data = np.concatenate([data1,data2])
        label1 = np.array(D1.targets).reshape((-1))
        label2 = np.array(D2.targets).reshape((-1))
        label = np.concatenate([label1,label2])
        return data.reshape((data.shape[0], -1))/ 255, label