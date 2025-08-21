import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from glob import glob

class KidneyDataset(Dataset):
    def __init__(self, FOLDER, used_classes=[0, 1, 2, 3], return_name=False):
        """
        :param FOLDER: directory to data .pt
        :param used_classes: list of classes used
        :param return_name: return data file name
        """
        self.paths = sorted(glob(os.path.join(FOLDER, "*.pt")))
        self.used_classes = used_classes
        self.return_name = return_name
        self.class_map = {cls: i for i, cls in enumerate(self.used_classes)}
        self.num_classes = len(self.used_classes)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = torch.load(path)
        image = data["image"]
        label = data["label"]

        remapped = torch.full_like(label, 0)
        for original_cls, new_cls in self.class_map.items():
            remapped[label == original_cls] = new_cls

        valid_mask = torch.zeros_like(label, dtype=torch.bool)
        for cls in self.used_classes:
            valid_mask |= (label == cls)
        remapped = remapped * valid_mask.long()
        
        remapped = remapped.long()
        label_onehot = F.one_hot(remapped, num_classes=self.num_classes).permute(3, 0, 1, 2).float()

        if self.return_name:
            return image, label_onehot, os.path.basename(path).replace(".pt", "")
        else:
            return image, label_onehot
