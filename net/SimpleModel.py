import torch
import torch.nn as nn
import logging
from .Convolution import Conv2d


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('SimpleModel')


def compose_cnn_simple():
    model = nn.Sequential(
        # block 1 - 3x3+NIN, in-50x50, out-25x25
        Conv2d(1, 32, 3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(32, 16, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(16, 16, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # block 2 - 3x3+NIN, in-25x25, out-12x12
        Conv2d(16, 128, 3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(128, 64, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(64, 64, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # block 3 - 3x3+NIN, in-12x12, out-6x6
        Conv2d(64, 256, 3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(256, 128, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(128, 128, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # block 4 - 3x3+NIN, in-6x6, out-3x3
        Conv2d(128, 512, 3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(512, 1024, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(1024, 256, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(256, 64, 1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # output layers - FC in-576, out-71
        nn.Flatten(),
        nn.Linear(in_features=576, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=71),
        # nn.Softmax()  # softmax disabled because of pytorch CrossEntropy Loss needs un-normalized output
    )
    return model


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = compose_cnn_simple()
        return

    def forward(self, inputs):
        out = self.cnn(inputs)
        return out
