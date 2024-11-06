from data_model.dataset_meta import DigitsDataset
from data_model.dataset_meta import DigitsSEQDataset
import numpy as np
class Newsgroups20Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
        vectorizer = TfidfVectorizer()
        data = vectorizer.fit_transform(newsgroups_train.data).todense()
        label = newsgroups_train.target
        print(data)
        print(data.shape)
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label).astype(np.int32)

        return data, label

class NG20Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        data=np.load(data_path+"/20NG.npy")
        label=np.load(data_path+"/20NG_labels.npy")
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        print(data)
        label = np.array(label).astype(np.int32)
        print(label)
        return data, label