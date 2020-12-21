import _pickle as pkl
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import preprocess
import os
import _pickle as pkl


class SpineDataset(Dataset):

    def __init__(
            self,
            path_to_images,
            fold,
            transform=None,
            sample=0):

        self.transform = transform
        filename = {'train': 'train.pkl', 'val': 'test.pkl'}
        with open(os.path.join(path_to_images, filename[fold]), "rb") as f:
            data, landmarks, labels = pkl.load(f)
        self.data = data
        self.landmarks = landmarks
        self.labels = labels
        
    
    def __len__(self):
        return self.data.shape[0]

    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return (image, self.labels[idx])
