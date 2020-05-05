# training dataset implement here
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import logging
from visualization import grid_plot
from config import *


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Dataset')


class TrainDataset(Dataset):
    def __init__(self, data_path='./dataset/data.npz', test_size=0.15, train_size=None, mode='train',
                 color=False, normalize=True):
        """
        This is the class for training dataset used by PyTorch DataLoader.
        Dataset depicts the logic of how to access each (image, label) pair.
        DataLoader depicts the logic of loading data during the training (batch size, shuffle, etc)

        :param data_path: str, the path to the dataset. Default to the local "./dataset" folder.
        :param test_size: float or int, If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
                                        If int, represents the absolute number of test samples.
        :param train_size: float or int,If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
                                        If int, represents the absolute number of train samples.
                                        If None, the value is automatically set to the complement of the test size.
        :param mode: str, 'train' or 'test', based on the mode, the dataset will produce different samples,
                                             before using the instance, use 'set_mode' function to set instance behavior
                                             you can not change mode during iterating the dataset.
        :param color: bool, if True returns 3 channels training images (converted from grayscale to RGB), default False.
        :param normalize: bool, if True, normalize the image from a range of [0,1] to a range of [-1, 1]
        """
        # load file
        file = np.load(data_path)

        # training data
        self.images = file['arr_0']  # shape (12211, 2500)
        self.labels = file['arr_1']  # shape (12211,)

        # attributes
        self.mode = mode
        self.color = color
        self.normalize = normalize
        self.c2l = None  # store class->label name lookup, e.g. {1:'a', 2:'ba'...}
        self.ci, self.cc = np.unique(self.labels, return_counts=True)  # classes statistics: unique labels, class count
        self.transform = self.compose_transform()  # a set of defined preprocessing techniques for the image
        self.trainset = {'image': None, 'label': None}  # training set after splitting
        self.testset = {'image': None, 'label': None}  # testing set after splitting

        # preprocessing
        self.label_to_classes()
        self.train_test_split(test_size, train_size)
        self.train_len = self.trainset['image'].shape[0]  # training set size
        self.test_len = self.testset['image'].shape[0]  # testing set size
        return

    def __len__(self):
        """
        Used by PyTorch DataLoader to know how many training data we have
        :return: total images number
        """
        if self.mode == 'train':
            return self.train_len
        elif self.mode == 'test':
            return self.test_len
        else:
            LOG.error("Unknown mode %s found." % self.mode)
            raise ValueError("Unknown mode %s found." % self.mode)

    def __getitem__(self, idx):
        """
        Used by PyTorch DataLoader to access the image, label pair at a specific index.
        :param idx: int, an index to specify which image to load.
        :return: pre-processed image and one-hot label
        """
        # decide which data to use based on the mode
        if self.mode == 'train':
            dataset = self.trainset
        elif self.mode == 'test':
            dataset = self.testset
        else:
            LOG.error("Unknown mode %s found." % self.mode)
            raise ValueError("Unknown mode %s found." % self.mode)

        # get label class
        label = dataset['label'][idx]

        # get training image and reshape
        image = dataset['image'][idx].reshape(50, 50)  # shape (H, W)

        # TODO: add more preprocessing techniques later here
        # convert to RGB if needed
        if self.color:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # shape (H, W, 3)

        # rescale to range [0, 1] then normalize to [-1, 1]
        image = image / 255.0  # to range [0, 1]
        if self.normalize:
            image = self.transform(image)

        return image, label

    def compose_transform(self):
        """
        A set of preprocessing to perform
        :return: a magic blackbox(pipeline) to perform the defined preprocessing techniques.
        """
        if self.color:
            return Compose([
                ToTensor(),  # convert ndarray to tensor -> shape (3, H, W)
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalize image from [0, 1] to [-1, 1]
            ])
        else:
            return Compose([
                ToTensor(),  # convert ndarray to tensor -> shape (1, H, W)
                Normalize(mean=(0.5,), std=(0.5,))  # normalize image from [0, 1] to [-1, 1]
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

    def train_test_split(self, test_size, train_size):
        """
        split training & testing data, splitting is performed on each class (each class has almost the same volume)
        :param test_size: float or int, see doc in the __init__ func.
        :param train_size: float or int, see doc in the __init__ func.
        :return: -
        """
        self.trainset['image'], self.testset['image'], self.trainset['label'], self.testset['label'] = \
            train_test_split(self.images, self.labels, test_size=test_size, train_size=train_size,
                             stratify=self.labels, random_state=SEED)
        return

    def train(self):
        """
        change dataset behavior to training mode
        :return: -
        """
        self.mode = 'train'
        return

    def eval(self):
        """
        change dataset behavior to testing mode
        :return: -
        """
        self.mode = 'test'
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
