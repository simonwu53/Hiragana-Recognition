# training dataset implement here
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from visualization import grid_plot
from config import *


class TrainDataset(Dataset):
    def __init__(self, data_path='./dataset/data.npz'):
        """
        This is the class for training dataset used by PyTorch DataLoader.
        Dataset depicts the logic of how to access each (image, label) pair.
        DataLoader depicts the logic of loading data during the training (batch size, shuffle, etc)

        :param data_path: str, the path to the dataset. Default to the local "./dataset" folder.
        """
        # load file
        file = np.load(data_path)

        # training data
        self.images = file['arr_0']  # shape (12211, 2500)
        self.labels = file['arr_1']  # shape (12211,)

        # attributes
        self.c2l = None  # store class->label name lookup, e.g. {1:'a', 2:'ba'...}
        self.len = self.images.shape[0]  # dataset volume
        self.ci, self.cc = np.unique(self.labels, return_counts=True)  # classes statistics: unique labels, class count
        self.transform = self.compose_transform()  # a set of defined preprocessing techniques for the image

        # preprocessing
        self.label_to_classes()
        return

    def __len__(self):
        """
        Used by PyTorch DataLoader to know how many training data we have
        :return: total images number
        """
        return self.len

    def __getitem__(self, idx):
        """
        Used by PyTorch DataLoader to access the image, label pair at a specific index.
        :param idx: int, an index to specify which image to load.
        :return: pre-processed image and one-hot label
        """
        # transform to one-hot label
        label = np.zeros((self.ci.shape[0],), dtype=np.float32)
        label[self.labels[idx]] = 1

        # normalize training image, TODO: add more preprocessing techniques later
        image = self.images[idx].reshape(50, 50)  # shape (H, W)
        image = image / 255.0  # to range [0, 1]
        image = self.transform(image)

        return image, label

    def compose_transform(self):
        """
        A set of preprocessing to perform
        :return: a magic blackbox(pipeline) to perform the defined preprocessing techniques.
        """
        return Compose([
            ToTensor(),  # convert ndarray to tensor -> shape (1, H, W)
            Normalize(mean=(TRAINSET_MEAN,), std=(TRAINSET_STD,))  # normalize image to [-1, 1]
        ])

    def label_to_classes(self):
        """
        convert unicode string label to integer classes,
        and prepare a dictionary for converting classes back to label string.
        :return: -
        """
        # get all unique labels
        labels = np.unique(self.labels)

        c2l = {}
        l2c = {}
        for i, l in enumerate(labels):
            c2l[i] = l.decode('utf-8')
            l2c[l.decode('utf-8')] = i

        self.labels = np.array(list(map(lambda x: l2c[x.decode('utf-8')], self.labels)))
        self.c2l = c2l
        return

    def show_random_example(self):
        """
        Plot a random sample for each class
        :return: -
        """
        # random choose one image from the category
        img_list = [self.images[np.random.choice(np.arange(self.images.shape[0])[self.labels==k], 1)[0]].reshape(50, 50) for k in self.c2l]
        labels = [self.c2l[k] for k in self.c2l]
        grid_plot(img_list, labels=labels)
        return
