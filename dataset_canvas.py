# training dataset implement here
from torch.utils.data import Dataset
import numpy as np
import logging
import random
from PIL import Image
import cv2
from lib import get_random_canvas, plot_one_box


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Dataset')


class TrainCanvasDataset(Dataset):
    def __init__(self, data_path='./dataset/data.npz', train_len=20000, test_len=3000, **kwargs):
        """
        This is the class for training dataset used by PyTorch DataLoader.
        Dataset depicts the logic of how to access each (image, label) pair.
        DataLoader depicts the logic of loading data during the training (batch size, shuffle, etc)

        Default params:
        :param data_path: str, the path to the dataset. Default to the local "./dataset" folder.
        :param test_len: int, set testing data length (volume)
        :param train_len: int, set training data length (volume)

        Optional params:
        :param max_characters: Optional[int], maximum number of characters per canvas
        :param min_characters: Optional[int], minimum number of characters per canvas
        :param max_scale: Optional[int], maximum character size in pixel, default size is 50x50
        :param min_scale: Optional[int], minimum character size in pixel, default size is 50x50
        :param color: bool, if True, return 3 channel RGB image
        """
        # load file
        file = np.load(data_path)

        # training data
        self.images = file['arr_0'].astype(np.uint8)  # shape (12211, 2500)
        self.images = self.images.reshape(
            self.images.shape[0], 50, 50)  # shape (12211, 50, 50)
        self.labels = file['arr_1']  # shape (12211,)

        # attributes list
        # dataset size
        self.train_len = train_len
        self.test_len = test_len
        # dataset mode
        self.mode = 'train'
        # store class->label name lookup, e.g. {1:'a', 2:'ba'...}
        self.c2l = None
        # store label->class name lookup, e.g. {'a':1, 'ba':2...}
        self.l2c = None
        # classes statistics: unique labels, class count
        self.unique_labels, self.class_count = np.unique(self.labels, return_counts=True)
        # convert str labels to int classes
        self.label_to_classes()

        # optional attributes
        self.min_characters = kwargs.get('min_characters', 3)
        self.max_characters = kwargs.get('max_characters', 10)
        LOG.warning('Characters per canvas: [%d, %d]' % (self.min_characters, self.max_characters))
        self.min_scale = kwargs.get('min_scale', 20)  # not used atm
        self.max_scale = kwargs.get('max_scale', 100)  # not used atm
        LOG.warning('Characters size range: [%d, %d] (not used)' % (self.min_scale, self.max_scale))
        self.color = kwargs.get('color', False)
        LOG.warning('Color mode %s.' % 'on' if self.color else 'off')
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
            LOG.error("Unknown mode '%s' found." % self.mode)
            raise ValueError("Unknown mode '%s' found." % self.mode)

    def __getitem__(self, idx):
        """
        Used by PyTorch DataLoader to access the image, label pair at a specific index.
        :param idx: int, an index to specify which image to load.
        :return: pre-processed image and one-hot label
        """
        # If we have made per_canvas static use that, otherwise get random from 3-10
        if self.min_characters != self.max_characters:
            per_canvas = random.randint(self.min_characters, self.max_characters)
        else:
            per_canvas = self.min_characters

        indices = np.random.choice(len(self.images), per_canvas)

        labels = np.take(self.labels, indices, axis=0)
        images = np.take(self.images, indices, axis=0)

        canvas, bboxes = get_random_canvas(images)

        # convert to color mode
        if self.color:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

        # convert to range [0, 1]
        canvas = canvas / 255.0

        return canvas, bboxes, labels

    def label_to_classes(self):
        """
        convert unicode string label to integer classes,
        and prepare a dictionary for converting classes back to label string.
        :return: -
        """
        LOG.warning("Converting labels to classes integers...")
        c2l = {}
        l2c = {}
        for i, l in enumerate(self.unique_labels):
            c2l[i] = l.decode('utf-8')
            l2c[l.decode('utf-8')] = i

        self.labels = np.array(
            list(map(lambda x: l2c[x.decode('utf-8')], self.labels)), dtype=np.int32)
        self.c2l = c2l
        self.l2c = l2c
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

    def show_random_example(self, show_labels=True):
        """
        Plot a random sample for each class

        :param show_labels: bool, if True, plot bounding boxes and its labels
        :return: -
        """
        # random choose one image from the category
        canvas, bboxes, labels = self.__getitem__(0)
        img = cv2.cvtColor((canvas*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if show_labels:
            colors = np.random.randint(0, 255, (len(bboxes), 3))
            for bbox, label, color in zip(bboxes, labels, colors):
                plot_one_box(bbox, img, color.tolist(), self.c2l[label])
        Image.fromarray(img).show()
        return
