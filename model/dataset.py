import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import pytorch_lightning as pl
import pandas as pd


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        batch_size: int,
        num_workers: int,
        train_val_ration: float,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_ration = train_val_ration

    def _get_dataset(self, path: str, target: str = "target"):
        df = pd.read_csv(path)
        X_tensor, y_tensor = torch.tensor(
            df.drop(columns=[target]).values, dtype=torch.float32
        ), torch.tensor(df[target].values, dtype=torch.float32)
        tensor_dataset = TensorDataset(X_tensor, y_tensor)
        return tensor_dataset

    def setup(self, stage):
        train_dataset = self._get_dataset(self.train_data_path)
        train_len = int(len(train_dataset) * self.train_val_ration)
        val_len = len(train_dataset) - train_len
        self.train_ds, self.val_ds = random_split(train_dataset, [train_len, val_len])

        test_dataset = self._get_dataset(self.test_data_path)
        self.test_ds = test_dataset

    def _get_data_loader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self._get_data_loader(self.train_ds, True)

    def val_dataloader(self):
        return self._get_data_loader(self.val_ds)

    def test_dataloader(self):
        return self._get_data_loader(self.test_ds)
