import torch
from lightning import pytorch as pl

from torchvision.datasets import SVHN
from torchvision import transforms

from torch.utils.data import DataLoader


class SVHNDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./data",
                 batch_size=1000,
                 num_workers=5) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                self._normalize(),
            ]
        )

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), self._normalize()]
        )

    def prepare_data(self):
        SVHN(self.data_dir, split="train", download=True)
        SVHN(self.data_dir, split="test", download=True)

    def setup(self, stage: str):
        self.svhn_train = SVHN(
            self.data_dir, split="train", transform=self.transform_train
        )

        self.svhn_test = SVHN(
            self.data_dir, split="test", transform=self.transform_test
        )

    def train_dataloader(self):
        return DataLoader(
            self.svhn_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.svhn_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.svhn_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.svhn_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _normalize():
        return transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]
        )
