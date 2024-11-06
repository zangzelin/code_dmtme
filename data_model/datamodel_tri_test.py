import lightning as pl
from data_model import dataset_all as dataset
from torch.utils.data import DataLoader


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_name: str = "Digits",
        data_path: str = "/zangzelin/data",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = False,
        pca_dim: int = 50,
        n_cluster: int = 25,
        n_f_per_cluster: int = 3,
        l_token: int = 10,
        seed: int = 0,
        rrc_rate: float = 0.8,
        trans_range: int = 6,
        sample_len: int = 500,
        num_positive_samples=1,
    ):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.n_cluster = n_cluster
        self.n_f_per_cluster = n_f_per_cluster
        self.l_token = l_token
        self.K = K
        self.seed = seed
        self.rrc_rate = rrc_rate
        self.trans_range = trans_range
        self.sample_len = sample_len
        self.num_positive_samples = num_positive_samples

    def setup(self, stage: str):
        dataset_meta = getattr(dataset, self.data_name + "Dataset")
        self.data_train = dataset_meta(
            data_name=self.data_name,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
        )
        self.data_val = dataset_meta(
            data_name=self.data_name,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
            train_val="val",
        )
        self.data_test = dataset_meta(
            data_name=self.data_name,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
            train_val="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        
        val1 = DataLoader(
            self.data_train,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        
        val2 = DataLoader(
            self.data_val,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_val.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        val3 = DataLoader(
            self.data_test,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_test.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return [val1, val2, val3]

    def test_dataloader(self):
        return DataLoader(
            self.data_val,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        
class MySEQDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_name: str = "Digits",
        data_path: str = "/zangzelin/data",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = False,
        pca_dim: int = 50,
        n_cluster: int = 25,
        n_f_per_cluster: int = 3,
        l_token: int = 10,
        seed: int = 0,
        rrc_rate: float = 0.8,
        trans_range: int = 6,
        sample_len: int = 500,
        num_positive_samples=1,
    ):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.n_cluster = n_cluster
        self.n_f_per_cluster = n_f_per_cluster
        self.l_token = l_token
        self.K = K
        self.seed = seed
        self.rrc_rate = rrc_rate
        self.trans_range = trans_range
        self.sample_len = sample_len
        self.num_positive_samples = num_positive_samples

    def setup(self, stage: str):
        dataset_meta = getattr(dataset, self.data_name + "SEQDataset")
        self.data_train = dataset_meta(
            data_name=self.data_name,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
        )
        self.data_val = dataset_meta(
            data_name=self.data_name,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
        )
        self.data_test = dataset_meta(
            data_name=self.data_name,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            uselabel=self.uselabel,
            uniform_param=1,
            num_positive_samples=self.num_positive_samples,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            drop_last=False,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )