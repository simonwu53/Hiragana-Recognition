# training dataset implement here
from torch.utils.data import IterableDataset, Dataset, DataLoader
import os
import numpy as np
from visualization import grid_plot


class TrainDataset(Dataset):
    def __init__(self):
        # load file
        file = np.load('./dataset/data.npz')

        # training data
        self.images = file['arr_0']
        self.labels = file['arr_1']

        # attributes
        self.c2l = None  # store class->label name lookup, e.g. {1:'a', 2:'ba'...}

        # preprocessing
        self.label_to_classes()
        return

    def __len__(self):
        return

    def __getitem__(self, idx):
        return

    def label_to_classes(self):
        # get all unique labels
        labels = np.unique(self.labels)

        c2l = {}
        l2c = {}
        for i, l in enumerate(labels):
            c2l[i] = l
            l2c[l] = i

        self.labels = np.array(list(map(lambda x: l2c[x], self.labels)))
        self.c2l = c2l
        return

    def show_random_example(self):
        # random choose one image from the category
        img_list = [self.images[np.random.choice(np.arange(self.images.shape[0])[self.labels==k], 1)[0]].reshape(50, 50) for k in self.c2l]
        labels = [self.c2l[k].decode('utf-8') for k in self.c2l]
        grid_plot(img_list, labels=labels)
        return
