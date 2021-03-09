import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch

from typing import Optional
from tqdm import tqdm


class CatDog(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame = None,
        file_list: List = None,
        directory: str = None,
        config: type = None,
        transforms: type = None,
        mode: str = "train",
    ) -> torch.tensor:

        self.df = df
        self.file_list = file_list
        self.directory = directory
        self.config = config
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        """Get the dataset length."""
        return len(self.df) if self.df is not None else len(self.file_list)

    def __getitem__(self, idx: int):
        """Get a row from the dataset."""

        if self.mode == "train":
            # special case, we create two datasets and concat.
            if "dog" in self.file_list[0]:
                label = 1
            else:
                label = 0
            label = torch.as_tensor(data=label, dtype=torch.int64, device=None)
            image_path = os.path.join(
                self.config.paths["train_path"], self.file_list[idx]
            )
        else:
            pass

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms.augment(image)
        else:
            image = torch.as_tensor(data=image, dtype=torch.float32, device=None)

        return image, label
