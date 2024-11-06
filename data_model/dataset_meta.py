from sklearn.datasets import load_digits

from torch.utils import data
import numpy as np
import os
from sklearn.decomposition import PCA
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import joblib


class DigitsDataset(data.Dataset):
    def __init__(
        self,
        data_name="Digits",
        data_path="/zangzelin/data",
        k=10,
        pca_dim=100,
        uselabel=False,
        uniform_param=1,
        train_val="train",
        split_ratio=0.8,
        split_ratio_test=0.1,
        num_positive_samples=1,
        use_neighbor=True,
        augment_bool=True,
    ):
        self.data_name = data_name
        data, label = self.load_data(data_path=data_path)
        
        index_save_path = f'data/'
        self.augment_bool = augment_bool
        

        train_data, train_label, val_data, val_label, test_data, test_label = self.split_data(
            data, label,
            split_ratio=split_ratio,
            split_ratio_test=split_ratio_test,
            train_val=train_val,
            data_name=data_name,
            index_save_path=index_save_path,
            )
        
        if data_name == 'Cifar10' or data_name == 'Cifar10T10':
            use_neighbor = False
    
        print('train_data.shape', train_data.shape, 
              'train_label.shape', train_label.shape, 
              'val_data.shape', val_data.shape, 
              'val_label.shape', val_label.shape,
              'test_data.shape', test_data.shape, 
              'test_label.shape', test_label.shape)
        
        print('train_val', train_val)
        
        self.train_val = train_val
        
        if train_val == 'train':
            self.data = train_data
            self.label = train_label
        elif train_val == 'val':
            self.data = val_data
            self.label = val_label
        elif train_val == 'test':
            self.data = test_data
            self.label = test_label
        
        self.uniform_param = uniform_param
        
        if use_neighbor:
            self.neighbors_index = self.Preprocessing(
                data, k, pca_dim, uselabel, train_val, split_ratio, split_ratio_test
            )
        
    def split_data(self, data, label, split_ratio=0.8, split_ratio_test=0.1, train_val='train', data_name='',index_save_path='data', index_save_name='train_val.npy'):

        data_len = data.shape[0]
        train_len = int(data_len * split_ratio)
        test_len = int(data_len * split_ratio_test)
        if not os.path.exists(index_save_path):
            os.makedirs(index_save_path)

        index_save_name_train = f'trainval_index_train_{split_ratio}_{split_ratio_test}_{data_name}.npy'
        index_save_name_val = f'trainval_index_val_{split_ratio}_{split_ratio_test}_{data_name}.npy'
        index_save_name_test = f'trainval_index_test_{split_ratio}_{split_ratio_test}_{data_name}.npy'
        if not os.path.exists(os.path.join(index_save_path, index_save_name_train)):
            print('save index', index_save_name_train, index_save_name_val, index_save_name_test)
            rand_index = np.random.permutation(data_len)
            train_index = rand_index[:train_len]
            val_index = rand_index[train_len: train_len + test_len]
            test_index = rand_index[train_len + test_len:]
            np.save(os.path.join(index_save_path, index_save_name_train), train_index)
            np.save(os.path.join(index_save_path, index_save_name_val), val_index)
            np.save(os.path.join(index_save_path, index_save_name_test), test_index)
        else:
            print('load index', index_save_name_train, index_save_name_val, index_save_name_test)
            train_index = np.load(os.path.join(index_save_path, index_save_name_train))
            val_index = np.load(os.path.join(index_save_path, index_save_name_val))
            test_index = np.load(os.path.join(index_save_path, index_save_name_test))
        train_data = data[train_index]
        train_label = label[train_index]
        val_data = data[val_index]
        val_label = label[val_index]
        test_data = data[test_index]
        test_label = label[test_index]

        return train_data, train_label, val_data, val_label , test_data, test_label

    def load_data(self, data_path):
        digit = load_digits()
        data = np.array(digit.data).astype(np.float32)
        label = np.array(digit.target)
        return data, label

    def Preprocessing(self, data, k, pca_dim, uselabel, token_index=0, train_val='train', split_ratio=0.8,split_ratio_test=0.1):
        self.graphwithpca = False

        neighbors_index = self.cal_near_index(
            k=k,
            uselabel=uselabel,
            pca_dim=pca_dim,
            train_val=train_val,
            split_ratio=split_ratio,
        )
        return neighbors_index

    def cal_near_index(self, k=10, device="cuda", uselabel=False, pca_dim=100, train_val='train', split_ratio=0.8):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}train_val{}split_ratio{}.pkl".format(
            self.data_name, k, uselabel, pca_dim, train_val, split_ratio
        )
        
        os.makedirs("save_near_index", exist_ok=True)
        if not os.path.exists(filename):
            X_rshaped = self.data.reshape((self.data.shape[0], -1))
            if pca_dim < X_rshaped.shape[1]:
                X_rshaped = PCA(n_components=pca_dim).fit_transform(X_rshaped)
            if not uselabel:
                index = NNDescent(X_rshaped, n_jobs=-1)
                neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
                neighbors_index = neighbors_index[:, 1:]
            else:
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(self.label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M - M.T) != 0] = dis.max() + 1
                neighbors_index = dis.argsort(axis=1)[:, 1 : k + 1]
            joblib.dump(value=neighbors_index, filename=filename)

            print("save data to ", filename)
        else:
            print("load data from ", filename)
            neighbors_index = joblib.load(filename)
        return neighbors_index

    def augment(self, data_input_item, index):
        
        # print('------', index)
        if self.augment_bool:
            neighbor_index_list = self.neighbors_index[index]
            selected_index = np.random.choice(neighbor_index_list)
            alpha = np.random.uniform(0, self.uniform_param)
            try:
                selected_data_input = self.data[selected_index]
                data_input_aug = data_input_item * alpha + selected_data_input * (1 - alpha)
            except:
                data_input_aug = data_input_item
        else:
            # print('no augment', 'self.augment_bool', self.augment_bool, )
            data_input_aug = data_input_item
        return data_input_item, data_input_aug

    def __getitem__(self, index):
        
        index = index % self.data.shape[0]

        # print('index', index)
        data_input_item = self.data[index]
        label = self.label[index]
        data_input_item, data_input_aug = self.augment(data_input_item, index)
        
        data_input_item = data_input_item.astype(np.float32)
        data_input_aug = data_input_aug.astype(np.float32)
        
        # print('data_input_item', data_input_item.shape, 'data_input_aug', data_input_aug.shape)
        return {
            "data_input_item": data_input_item,
            "data_input_aug": data_input_aug,
            "label": label,
            "index": index,
        }

    def __len__(self):
        if self.data_name == 'Cifar10' or self.data_name == 'Cifar100':
            return self.data.shape[0] * 10
        else:
            return self.data.shape[0]



class DigitsSEQDataset(data.Dataset):
    def __init__(
        self,
        data_name="Digits",
        data_path="/zangzelin/data",
        k=10,
        pca_dim=100,
        seq_len=500,
        uselabel=False,
        uniform_param=1,
        num_positive_samples=1,
    ):
        self.data_name = data_name
        self.seq_len = seq_len
        self.data, self.label = self.load_data(data_path=data_path)
        self.uniform_param = uniform_param
        self.nps = num_positive_samples
        
        self.neighbors_index = self.Preprocessing(
            data, k, pca_dim, uselabel
        )

    def load_data(self, data_path):
        digit = load_digits()
        data = np.array(digit.data).astype(np.float32)
        label = np.array(digit.target)
        return data, label

    def Preprocessing(self, data, k, pca_dim, uselabel, token_index=0):
        self.graphwithpca = False

        neighbors_index = self.cal_near_index(
            k=k,
            uselabel=uselabel,
            pca_dim=pca_dim,
        )
        return neighbors_index

    def cal_near_index(self, k=10, device="cuda", uselabel=False, pca_dim=100):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.data_name, k, uselabel, pca_dim
        )
        os.makedirs("save_near_index", exist_ok=True)
        
        if not os.path.exists(filename):
            X_rshaped = self.data.reshape((self.data.shape[0], -1))
            if pca_dim < X_rshaped.shape[1]:
                X_rshaped = PCA(n_components=pca_dim).fit_transform(X_rshaped)
            if not uselabel:
                index = NNDescent(X_rshaped, n_jobs=-1)
                neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
                neighbors_index = neighbors_index[:, 1:]
            else:
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(self.label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M - M.T) != 0] = dis.max() + 1
                neighbors_index = dis.argsort(axis=1)[:, 1 : k + 1]
            joblib.dump(value=neighbors_index, filename=filename)

            print("save data to ", filename)
        else:
            print("load data from ", filename)
            neighbors_index = joblib.load(filename)
        return neighbors_index

    def feature_to_idex(self, feature, seq_len):
        
        # import pdb; pdb.set_trace()
        # print(feature)
        rank_input = np.argsort(feature)[::-1] [: seq_len]
        # print(rank_input)
        # sample_index = np.random.choice(rank_input.shape[0], seq_len, replace=False)
        # mask = np.zeros(rank_input.shape[0])
        # mask[sample_index] = 1
        # rank_input = rank_input[mask == 1]
        
        return rank_input

    def get_augment(self, data_input_item, neighbor_index_list):
        selected_index = np.random.choice(neighbor_index_list)
        alpha = np.random.uniform(0, self.uniform_param)
        selected_data_input = self.data[selected_index]
        data_input_aug = data_input_item * alpha + selected_data_input * (1 - alpha)
        return data_input_aug.reshape(1, -1)        

    def __getitem__(self, index):

        if self.train_val == 'train':
            data_input_item = self.data[index].astype(np.float32)
            label = self.label[index]
            neighbor_index_list = self.neighbors_index[index]

            data_input_aug = []
            for i in range(self.nps):
                data_input_aug.append(
                    self.get_augment(data_input_item, neighbor_index_list))
            
            data_input_aug = np.concatenate(data_input_aug, axis=0)

            rank_input = self.feature_to_idex(data_input_item, seq_len=self.seq_len)
            rank_aug = self.feature_to_idex(data_input_aug, seq_len=self.seq_len)

            return{
                "rank_input": rank_input.astype(np.int64),
                "rank_aug": rank_aug.astype(np.int64),
                "data_input_item": data_input_item,
                "data_input_aug": data_input_aug,
                "label": label,
                "index": index,
            }
        else:
            print('test')
            data_input_item = self.data[index].astype(np.float32)
            label = self.label[index]
            neighbor_index_list = self.neighbors_index[index]

            data_input_aug = []
            for i in range(self.nps):
                data_input_aug.append(
                    self.transform_test(data_input_item)
                    )
            
            data_input_aug = np.concatenate(data_input_aug, axis=0)

            rank_input = self.feature_to_idex(data_input_item, seq_len=self.seq_len)
            rank_aug = self.feature_to_idex(data_input_aug, seq_len=self.seq_len)

            return{
                "rank_input": rank_input.astype(np.int64),
                "rank_aug": rank_aug.astype(np.int64),
                "data_input_item": data_input_item,
                "data_input_aug": data_input_aug,
                "label": label,
                "index": index,
            }

    def __len__(self):
        return self.data.shape[0]