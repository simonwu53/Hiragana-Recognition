# training dataset implement here
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import logging
import random
from visualization import grid_plot
from config import *
from lib import get_random_canvas


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Dataset')


class TrainCanvasDataset(Dataset):
    def __init__(self, data_path='./dataset/data.npz', test_size=0.15, train_size=None, mode='train',
                 img_size=50, per_canvas=None, train_len=20000, test_len=3000):
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
        :param img_size: int, specify the image shape after pre-processing, default size is 50
        :param normalize: bool, if True, normalize the image from a range of [0,1] to a range of [-1, 1]
        """
        # load file
        file = np.load(data_path)

        # training data
        self.images = file['arr_0'].astype(np.uint8)  # shape (12211, 2500)
        self.images = self.images.reshape(
            self.images.shape[0], 50, 50)  # shape (12211, 50, 50)
        self.labels = file['arr_1']  # shape (12211,)

        # attributes
        self.mode = mode
        # store class->label name lookup, e.g. {1:'a', 2:'ba'...}
        self.c2l = None
        # classes statistics: unique labels, class count
        self.ci, self.cc = np.unique(self.labels, return_counts=True)
        # a set of defined preprocessing techniques for the image
        self.transform = self.compose_transform()
        self.per_canvas = per_canvas
        self.label_to_classes()
        self.train_len = train_len
        self.test_len = test_len

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

    def __getitem__(self):
        """
        Used by PyTorch DataLoader to access the image, label pair at a specific index.
        :param idx: int, an index to specify which image to load.
        :return: pre-processed image and one-hot label
        """
        # If we have made per_canvas static use that, otherwise get random from 3-10
        per_canvas = self.per_canvas
        if per_canvas == None:
            per_canvas = random.randint(3, 10)

        indices = np.random.choice(len(self.images), per_canvas)

        labels = np.take(self.labels, indices, axis=0)
        images = np.take(self.images, indices, axis=0)

        canvas, bboxes = get_random_canvas(images)
        canvas = canvas / 255.0

        return canvas, bboxes, labels

    def set_length(self, train_len, test_len):
        self.train_len = train_len
        self.test_len = test_len

    def compose_transform(self):
        """
        A set of preprocessing to perform
        :return: a magic blackbox(pipeline) to perform the defined preprocessing techniques.
        """
        return Compose([
            ToTensor(),  # convert ndarray to tensor -> shape (1, H, W)
            # normalize to 0-mean, std at 1
            Normalize(mean=(MEAN,), std=(STD,))
        ])

    def label_to_classes(self):
        """
        convert unicode string label to integer classes,
        and prepare a dictionary for converting classes back to label string.
        :return: -
        """
        LOG.warning("Converting labels to classes integers...")
        # get all unique labels
        labels = np.unique(self.labels)

        c2l = {}
        l2c = {}
        for i, l in enumerate(labels):
            c2l[i] = l.decode('utf-8')
            l2c[l.decode('utf-8')] = i

        self.labels = np.array(
            list(map(lambda x: l2c[x.decode('utf-8')], self.labels)), dtype=np.int32)
        self.c2l = c2l
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
        canvas, bboxes, labels = self.__getitem__()
        cv2.imshow("canvas", canvas)
        return
