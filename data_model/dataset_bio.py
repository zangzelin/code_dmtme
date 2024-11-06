from sklearn.datasets import load_digits

from torch.utils import data
from sklearn.datasets import load_digits
from torch import tensor
import torchvision.datasets as datasets
from pynndescent import NNDescent
import os
import joblib
import torch
import numpy as np
from PIL import Image
import scanpy as sc
import scipy
from sklearn.decomposition import PCA
import sklearn
import pandas as pd

from data_model.dataset_meta import DigitsDataset
from data_model.dataset_meta import DigitsSEQDataset
import torchvision.datasets as datasets

np.random.seed(42)
seed=42
class MCAD9119Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        data = np.load(data_path + '/mca_data/mca_data_dim_34947.npy')
        data = data[:, data.max(axis=0) > 4].astype(np.float32)
        label = np.load(data_path + '/mca_data/mca_label_dim_34947.npy')

        label_count = {}
        for i in label:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1

        for i in list(label_count.keys()):
            if label_count[i] < 500:
                label[label == i] = -1 

        data = data[label != -1]
        label = label[label != -1]

        return data, label.astype(np.int32)

class ActivityDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_data = pd.read_csv(data_path+'/feature_select/Activity_train.csv')
        test_data = pd.read_csv(data_path+'/feature_select/Activity_test.csv')
        all_data = pd.concat([train_data, test_data])
        data = all_data.drop(['subject', 'Activity'], axis=1).to_numpy()
        label_str = all_data['Activity'].tolist()
        label_str_set = sorted(list(set(label_str)))
        label = np.array([label_str_set.index(i) for i in label_str])
        data = (data-data.min())/(data.max()-data.min())
        
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print('data.shape', data.shape)
        print(label)
        return data, label.astype(np.int32)

class Gast10k1458Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        
        sadata = sc.read(data_path+"/gast10kwithcelltype.h5ad")
        sadata_pca = np.array(sadata.X)
        data = np.array(sadata_pca).astype(np.float32)

        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)  # Create a column of zeros
        data = np.hstack((data, zeros_column))

        label_train_str = list(sadata.obs['celltype'])
        label_train_str_set = sorted(list(set(label_train_str)))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str])
        label = np.array(label_train).astype(np.int32)
        return data, label

class SAMUSIKDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        data = pd.read_csv(data_path+'/samusik_01.csv')
        label_str = pd.read_csv(data_path+'/samusik01_labelnr.csv')
        data.fillna(data.min(), inplace=True)
        label = np.array(label_str)[:,-1]
        data = np.array(data)[:,1:]
        data = (data-data.min())/(data.max()-data.min())
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label).astype(np.int32)
        # import pdb; pdb.set_trace()
        return data, label

class HCL60KDataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))
        
        # import pdb; pdb.set_trace()

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(np.bool_)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]
        
        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)  # Create a column of zeros
        data = np.hstack((data, zeros_column))
        tissue = list(sadata.obs['tissue'])
        
        
        # dict_label_tissue = 
        
        print(data.shape)
        return data, label



class HCL60KPLOTDataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        # label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))
        label_str = list(sadata.obs.tissue)
        set_list = list(set(label_str))
        label = np.array([set_list.index(i) for i in label_str])
        import pickle
        pickle.dump(set_list, open('HCL_set_list_label.pkl', 'wb'))
        
        
        # import pdb; pdb.set_trace()

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(np.bool_)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]
        
        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)  # Create a column of zeros
        data = np.hstack((data, zeros_column))
        tissue = list(sadata.obs['tissue'])
        
        
        # dict_label_tissue = 
        
        print(data.shape)
        return data, label

class HCL600KDataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(np.bool_)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]
        
        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)  # Create a column of zeros
        data = np.hstack((data, zeros_column))

        # repeat the data 10 times
        data = np.repeat(data, 10, axis=0)
        label = np.repeat(label, 10, axis=0)

        print(data.shape)
        return data, label


class HCL60K1000Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(np.bool_)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]
        
        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        feature_variances = np.var(data, axis=0)
        top_features_indices = np.argsort(feature_variances)[-1000:]
        data = data[:, top_features_indices]

        print(data.shape)
        return data, label

class CeleganDataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        adata = sc.read(data_path+"/celegan/celegan.h5ad")
        data = adata.X
        data = np.array(data).astype(np.float32)
        label_celltype = pd.read_csv(data_path+'/celegan/celegan_celltype_2.tsv', sep='\t', header=None)
        adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype))
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = sorted(list(set(label_train_str)))
        label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        print(data)
        return data, label