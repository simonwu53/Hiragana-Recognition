# training loop implement here
# ---libs---
import argparse
import torch
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
from datetime import datetime
import shutil
import logging
from config import *


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Train')


def train(args):
    """
    Training process main function

    :param args: args from command line inputs
    :return: -
    """
    # create training directory
    save_root = os.path.join('./results', datetime.now().strftime("%H%M_%d%m%Y"))
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
    global_i = 0  # global steps counter
    best_eval = 10000  # store the best validation result
    res = -1  # store the validation result of current epoch

    # dataset and loader
    dataset = TrainDataset(data_path=args.dataset, test_size=testSize, train_size=trainSize)
    loader = DataLoader(dataset, batch_size=BS, shuffle=SF, num_workers=numWorkers,
                        pin_memory=pinMem, drop_last=dropLast, timeout=timeOut)

    # continue training check
    if args.load:
        trained_epoch, global_i = load_ckpt(model, optimizer, args.load)
    else:
        trained_epoch = 0

    # start training epoch
    for epoch in range(trained_epoch, trained_epoch+Epochs):
        model.train()  # switch model to training mode
        dataset.train()  # switch dataset to training mode
        running_loss = 0.0  # running loss for training set
        running_loss_eval = 0.0  # running loss for testing set
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

        # validation
        model.eval()  # switch model to validation mode
        dataset.eval()  # switch dataset to validation mode

        with torch.no_grad():  # no need to track computation graph during testing, save resources and speedups
            LOG.warning('Evaluation on testing data...')
            for i, data in enumerate(tqdm(loader)):
                img, label = data[0].cuda(), data[1].cuda()
                logits = model(img)
                loss = loss_batch(logits, label, criteria)
                running_loss_eval += loss.item()

        # validation results
        res = running_loss_eval / len(loader)
        writer.add_scalar('Test/Loss', res, global_i)
        writer.flush()
        LOG.warning('Epoch %d: validation loss: %.4f' % (epoch + 1, res))

        # store the best model
        if res < best_eval:
            best_eval = res
            save_ckpt(model, optimizer, (epoch + 1), global_i,
                      os.path.join(ckpt_dir, 'Epoch%dloss%.4f.tar' % (epoch + 1, res)))

    # save the trained model
    save_ckpt(model, optimizer, trained_epoch+Epochs, global_i,
              os.path.join(ckpt_dir, 'Epoch%dloss%.4f.tar' % (trained_epoch+Epochs, res)))
    return


def test(args):
    return


def loss_batch(logits, gt, func, opt=None):
    """
    calculate the losses from the model's output and the ground truth
    :param logits: model output, must be compatible with ground truth and loss function
    :param gt: ground truth, must be compatible with logits and loss function
    :param func: loss function, must be compatible with logits and ground truth
    :param opt: optimizer instance
    :return: loss value
    """
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
    """
    return the corresponding model based on the command line args
    :param args: command line args
    :return:
    """
    if args.vgg:
        return models.vgg16()
    elif args.inception:
        return models.inception_v3()
    elif args.simple:
        return SimpleModel.CNN()
    else:
        LOG.error("You must select a model to train.")
        raise ValueError("No model selected. use --vgg or --inception")


def save_ckpt(model, optimizer, epoch, global_step, path):
    """
    Save the trained model checkpoint with a given name

    :param model: pytorch model to save
    :param optimizer: optimizer to save
    :param epoch: current epoch value
    :param global_step: current global step for tensorboard
    :param path: model path to save
    """
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    return


def load_ckpt(model, optimizer, path):
    """
    Load a pre-trained model on GPU for training or evaluation

    :param model: pytorch model object to load trained parameters
    :param optimizer: optimizer object used in the last training
    :param path: path to the saved checkpoint
    """
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt['epoch']
    global_step = ckpt['global_step']
    return epoch, global_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='', help="path to the saved 'tar' checkpoint file.")
    parser.add_argument('--dataset', type=str, default='./dataset/data.npz', help="Path to the dataset.")
    parser.add_argument('--train', action='store_true', help='Start training process.')
    parser.add_argument('--test', action='store_true', help='Start testing process.')
    parser.add_argument('--vgg', action='store_true', help='Use VGG Net.')
    parser.add_argument('--simple', action='store_true', help='Use simple CNN.')
    parser.add_argument('--inception', action='store_true', help='Use Inception V3 Net.')
    opt = parser.parse_args()
    print(opt)

    if opt.train:
        train(opt)
    elif opt.test:
        test(opt)
        raise NotImplementedError('testing mode is not implemented')
    else:
        LOG.warning("Please specify whether to train or test")
        raise ValueError("Please specify whether to train or test using --train or --test")
