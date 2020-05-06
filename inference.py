# training loop implement here
# ---libs---
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import make_grid
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
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
from lib import sample_images, count_parameters


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
    # save configuration to the training directory
    shutil.copy2('./config.py', save_root)

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
    graph_loaded = False  # tensorboard graph loading status
    n_samples = 500  # samples of features to project in tensorboard

    # dataset and loader
    if args.vgg or args.inception:
        dataset = TrainDataset(data_path=args.dataset, test_size=testSize, train_size=trainSize,
                               color=True, img_size=upSampling)
    else:
        dataset = TrainDataset(data_path=args.dataset, test_size=testSize, train_size=trainSize,
                               color=False, img_size=upSampling)
    loader = DataLoader(dataset, batch_size=BS, shuffle=SF, num_workers=numWorkers,
                        pin_memory=pinMem, drop_last=dropLast, timeout=timeOut)
    if len(loader) < 1:
        LOG.error('Dataset maybe empty.')
        raise ValueError('Dataset maybe empty.')

    # continue training check
    if args.load:
        trained_epoch, global_i = load_ckpt(model, optimizer, args.load)
    else:
        trained_epoch = 0

    # sample images and show the overview of the input features
    sampled_images, sampled_labels = sample_images(dataset.trainset['image'], dataset.trainset['label'], n_samples)
    writer.add_embedding(sampled_images.reshape(n_samples, -1),
                         metadata=[dataset.c2l[l] for l in sampled_labels],
                         label_img=torch.from_numpy(sampled_images[:, np.newaxis, :, :]))

    # start training epoch
    for epoch in range(trained_epoch, trained_epoch+Epochs):
        model.train()  # switch model to training mode
        dataset.train()  # switch dataset to training mode
        running_loss = 0.0  # running loss for training set
        running_acc = 0.0  # running accuracy for training set
        running_loss_eval = 0.0  # running loss for validation set
        running_acc_eval = 0.0  # running accuracy for validation set
        print()
        LOG.warning('Start epoch %d.' % (epoch + 1))

        # iterating each batch
        for i, data in enumerate(tqdm(loader)):
            img, label = data[0].float().cuda(), data[1].long().cuda()  # push the data to GPU
            pred = model(img)  # model inference

            # loss for one output and tuple output
            if opt.inception:
                loss = loss_batch(pred, label, criteria, optimizer, mode='inception')
                acc = (pred[0].argmax(1) == label).float().sum() / BS
            else:
                loss = loss_batch(pred, label, criteria, optimizer)
                acc = (pred.argmax(1) == label).float().sum() / BS

            # collect statistics
            running_loss += loss.item()
            running_acc += acc  # accuracy of this batch

            # update training statistics
            if i % TBUpdate == 0:
                if i == 0:
                    img_grid = make_grid(img)
                    writer.add_image('Train/Batch', img_grid, global_i)
                    if not graph_loaded:  # only add once
                        writer.add_graph(model, img)
                        graph_loaded = True
                writer.add_scalar('Train/Loss', loss.item(), global_i)  # or optimizer, dropout info
                writer.add_scalar('Train/Accuracy', acc, global_i)
                writer.flush()

            # update global step
            global_i += 1

        # show epoch info
        LOG.warning('Epoch %d: running loss: %.4f  running accuracy: %.2f' %
                    (epoch + 1, running_loss / len(loader), running_acc / len(loader)))

        # validation
        model.eval()  # switch model to validation mode
        dataset.eval()  # switch dataset to validation mode

        with torch.no_grad():  # no need to track computation graph during testing, save resources and speedups
            LOG.warning('Evaluation on testing data...')
            for i, data in enumerate(tqdm(loader)):
                img, label = data[0].float().cuda(), data[1].long().cuda()
                pred = model(img)
                loss = loss_batch(pred, label, criteria)
                acc = (pred.argmax(1) == label).float().sum() / BS
                running_loss_eval += loss.item()
                running_acc_eval += acc
                # validation batch for visualization
                if i == 0:
                    img_grid = make_grid(img)
                    writer.add_image('Validation/Batch', img_grid, global_i)

        # validation results
        res = running_loss_eval / len(loader)
        acc_eval = running_acc_eval / len(loader)
        writer.add_scalar('Validation/Loss', res, global_i)
        writer.add_scalar('Validation/Accuracy', acc_eval, global_i)
        writer.flush()
        LOG.warning('Epoch %d: validation loss: %.4f  accuracy: %.2f' % (epoch + 1, res, acc_eval))

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


def loss_batch(pred, gt, func, optimizer=None, mode='normal'):
    """
    calculate the losses from the model's output and the ground truth
    :param pred: model output, must be compatible with ground truth and loss function
    :param gt: ground truth, must be compatible with logits and loss function
    :param func: loss function, must be compatible with logits and ground truth
    :param optimizer: optimizer instance
    :param mode: str, 'normal' or 'inception', inception has additional auxiliary logits
    :return: loss value
    """
    if mode == 'normal':
        loss = func(pred, gt)
    elif mode == 'inception':
        logits, aux_logits = pred
        l1, l2 = func(logits, gt), func(aux_logits, gt)
        loss = l1 + 0.3 * l2
    else:
        LOG.error("Unknown model output? Need a method to compute loss...")
        raise ValueError("Unknown model output? Need a method to compute loss...")

    if optimizer is not None:
        # auto-calculate gradients
        loss.backward()
        # apply gradients
        optimizer.step()
        # zero the parameter gradients
        optimizer.zero_grad()
    return loss


def select_model(args):
    """
    return the corresponding model based on the command line args
    :param args: command line args
    :return:
    """
    # use the base vgg19 with batch normalization model
    if args.vgg:
        if upSampling < 224:
            LOG.error("Minimum input size for VGG net is 224!")
            raise ValueError("Minimum input size for VGG net is 224!")
        LOG.warning('Loading VGG19 with Batch Normalization model...')
        LOG.warning('It may take few minutes to load the PyTorch model...please wait patiently...')
        model = models.vgg19_bn()  # 5 maxpooling layers
        # modify the model classifier to match our dataset
        model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096),  # in 256*3*3, out 1024
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),  # keep the same dropout rate
                                         nn.Linear(in_features=4096, out_features=768),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(in_features=768, out_features=71))  # output 71 classes
        LOG.warning('Trainable parameters: %d' % count_parameters(model))
        return model

    # load inception v3 model
    elif args.inception:
        if upSampling < 299 and BS < 2:
            LOG.error("Minimum input size for Inception v3 net is 299, batch size must > 1!")
            raise ValueError("Minimum input size for Inception v3 net is 299, batch size must > 1!")
        LOG.warning('Loading Inception v3 model...')
        LOG.warning('It may take few minutes to load the PyTorch model...please wait patiently...')
        model = models.inception_v3(num_classes=71)
        LOG.warning('Trainable parameters: %d' % count_parameters(model))
        return model

    # load customzied model
    elif args.simple:
        model = SimpleModel.CNN()
        LOG.warning('Trainable parameters: %d' % count_parameters(model))
        return model
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
    parser.add_argument('--vgg', action='store_true', help='Use VGG 19 with Batch Normalization.')
    parser.add_argument('--simple', action='store_true', help='Use customized CNN.')
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
