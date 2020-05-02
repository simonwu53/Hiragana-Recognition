# training loop implement here
# ---libs---
import argparse
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# ---modules---
from dataset import TrainDataset
from net import SimpleModel
# ---misc---
from tqdm import tqdm
import os
import shutil
import logging
from config import *


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Train')


def train(args):
    # create training directory
    save_root = os.path.join('./results', args.name)
    if os.path.exists(save_root):
        LOG.warning('Output folder already exists, cleaning before continuing...')
        shutil.rmtree(save_root)  # delete output folder if exists!!!
    os.makedirs(save_root)
    log_dir = os.path.join(save_root, 'log/')
    os.mkdir(log_dir)
    ckpt_dir = os.path.join(save_root, 'checkpoints/')
    os.mkdir(ckpt_dir)

    # TensorBoard summary writer -- for training process visualization
    writer = SummaryWriter(log_dir=log_dir, max_queue=10, flush_secs=120)

    # get CNN model
    model = select_model(args)
    model.cuda()

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2, amsgrad=False)

    # loss function
    criteria = nn.CrossEntropyLoss(reduction='sum')

    # variables
    global_i = 0

    # dataset and loader
    dataset = TrainDataset(data_path=args.dataset)
    loader = DataLoader(dataset, batch_size=BS, shuffle=SF, num_workers=numWorkers,
                        pin_memory=pinMem, drop_last=dropLast, timeout=timeOut)

    # start training epoch
    for epoch in range(Epochs):
        model.train()  # switch to training mode
        running_loss = 0.0
        LOG.warning('Start epoch %d.' % (epoch + 1))

        # iterating each batch
        for i, data in enumerate(tqdm(loader)):
            img, label = data[0].cuda(), data[1].cuda()  # push the data to GPU

            logits = model(img)  # model inference
            loss = loss_batch(logits, label, criteria, optimizer)

            # collect statistics
            running_loss += loss.item()

            # update training statistics
            if i % TBUpdate == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_i)  # or optimizer, dropout info
                writer.flush()

            # update global step
            global_i += 1

        # show epoch info
        LOG.warning('Epoch %d: running loss: %.4f' % (epoch + 1, running_loss / len(loader)))
    return


def loss_batch(logits, gt, func, opt=None):
    loss = func(logits, gt)

    if opt is not None:
        # auto-calculate gradients
        loss.backward()
        # apply gradients
        opt.step()
        # zero the parameter gradients
        opt.zero_grad()
    return loss


def select_model(args):
    if args.vgg:
        return models.vgg16()
    elif args.inception:
        return models.inception_v3()
    elif args.simple:
        return SimpleModel.CNN()
    else:
        LOG.error("You must select a model to train.")
        raise ValueError("No model selected. use --vgg or --inception")


def save_ckpt():
    return


def load_ckpt():
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="test1",
                        help="Name of the testing. Corresponding folder will be created in './results'")
    parser.add_argument('--load', type=str, default="test1",
                        help="Name of the testing to load. Corresponding folder can be found in './results'")
    parser.add_argument('--dataset', type=str, default='./dataset/data.npz', help="Path to the dataset.")
    parser.add_argument('--train', action='store_true', help='Start training process.')
    parser.add_argument('--test', action='store_true', help='Start testing process.')
    parser.add_argument('--vgg', action='store_true', help='Use VGG Net.')
    parser.add_argument('--simple', action='store_true', help='Use simple CNN.')
    parser.add_argument('--inception', action='store_true', help='Use Inception V3 Net.')
    opt = parser.parse_args()
    print(opt)

    train(opt)
