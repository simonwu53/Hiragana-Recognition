# training loop implement here
# ---libs---
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import nms
import numpy as np
from PIL import Image
import cv2
# ---modules---
from dataset.dataset_canvas import TrainCanvasDataset, dataset_collate_fn
# ---model---
from net.fasterRCNN import fasterRCNN_ResNet50_fpn
# ---misc---
from tqdm import tqdm
import os
from datetime import datetime
import shutil
import logging
from config import *
from lib import count_parameters
from visualization import plot_one_box


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
    # get CNN model
    model = select_model(args)
    model.cuda()

    # get the params that needs to be trained only
    params = [p for p in model.parameters() if p.requires_grad]
    # create optimizer
    optimizer = optim.Adam(params, lr=LR, weight_decay=L2, amsgrad=False)

    # variables
    global_i = 0  # global steps counter
    best_loss = 10000

    # dataset and loader
    dataset, loader = load_dataset(args)
    if len(loader) < 1:
        LOG.error('Dataset maybe empty.')
        raise ValueError('Dataset maybe empty.')

    # continue training check & prepare saving directory
    if args.load:
        trained_epoch, global_i = load_ckpt(args.load, model, optimizer)
        save_root = os.path.dirname(os.path.dirname(args.load))
        log_dir = os.path.join(save_root, 'log/')
        ckpt_dir = os.path.join(save_root, 'checkpoints/')
        writer = SummaryWriter(log_dir=log_dir, purge_step=global_i, max_queue=10, flush_secs=120)
    else:
        trained_epoch = 0
        save_root = os.path.join('./results', datetime.now().strftime("%H%M_%d%m%Y"))
        os.mkdir(save_root)
        log_dir = os.path.join(save_root, 'log/')
        os.mkdir(log_dir)
        ckpt_dir = os.path.join(save_root, 'checkpoints/')
        os.mkdir(ckpt_dir)
        # save configuration to the training directory
        shutil.copy2('./config.py', save_root)
        # TensorBoard summary writer -- for training process visualization
        writer = SummaryWriter(log_dir=log_dir, max_queue=10, flush_secs=120)

    # start training epoch
    for epoch in range(trained_epoch, trained_epoch+Epochs):
        model.train()  # switch model to training mode
        dataset.train()  # switch dataset to training mode
        running_loss = 0.0  # running loss for training set
        print()
        LOG.warning('Start epoch %d.' % (epoch + 1))

        # iterating each batch
        for i, data in enumerate(tqdm(loader)):
            images, targets = data
            # push the data to GPU
            # images:FloatTensor[N, 4]
            images = [torch.from_numpy(image).float().cuda() for image in images]
            # 'boxes':Int64Tensor[N], 'labels':Int64Tensor[N]
            targets = [{k: torch.from_numpy(v).float().cuda() if k=='boxes' else torch.LongTensor(v).cuda() for k, v in t.items()} for t in targets]

            # model inference
            loss_dict = model(images, targets)

            # compute loss summation
            # dict_keys(['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'])
            losses = sum(loss for loss in loss_dict.values())
            # loss value only
            loss_value = losses.item()

            # zero the parameter gradients
            optimizer.zero_grad()
            # auto-calculate gradients
            losses.backward()
            # apply gradients
            optimizer.step()

            # collect statistics
            running_loss += loss_value

            # update training statistics
            if i % TBUpdate == 0:
                if i == 0:
                    for k, image in enumerate(images):
                        writer.add_image('Train/Batch%d' % k, image, global_i)
                rpn_objectness = loss_dict['loss_objectness'].item()
                rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()
                roi_classifier = loss_dict['loss_classifier'].item()
                roi_box_reg = loss_dict['loss_box_reg'].item()
                writer.add_scalar('Train/Loss_sum', loss_value, global_i)
                writer.add_scalar('Train/Loss_objectness', rpn_objectness, global_i)
                writer.add_scalar('Train/Loss_rpn_box_reg', rpn_box_reg, global_i)
                writer.add_scalar('Train/Loss_classifier', roi_classifier, global_i)
                writer.add_scalar('Train/Loss_roi_box_reg', roi_box_reg, global_i)
                writer.flush()

            # update global step
            global_i += 1

        # show epoch info
        epoch_loss = running_loss / len(loader)
        LOG.warning('Epoch %d: running loss: %.4f' % (epoch + 1, epoch_loss))

        # save the best training model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_ckpt(model, optimizer, (epoch + 1), global_i,
                      os.path.join(ckpt_dir, 'Epoch%dLoss%.2f.tar' % (epoch + 1, best_loss)))

        # validation
        model.eval()  # switch model to validation mode
        dataset.eval()  # switch dataset to validation mode

        with torch.no_grad():  # no need to track computation graph during testing, save resources and speedups
            LOG.warning('Evaluation on testing data...')
            for i, data in enumerate(tqdm(loader)):
                images, targets = data
                images_torch = [torch.from_numpy(image).float().cuda() for image in images]

                # model inference
                pred = model(images_torch)

                # interpret the predictions
                interpreted_images = interpret_predictions(images, targets, pred, dataset.c2l)

                # show batch in TensorBoard
                for j, plotted_img in enumerate(interpreted_images):
                    writer.add_image('Validation/Results%d_%d' % (i, j), torch.from_numpy(plotted_img.transpose(2, 0, 1)), global_i)
                writer.flush()

    # save the trained model
    save_ckpt(model, optimizer, trained_epoch+Epochs, global_i,
              os.path.join(ckpt_dir, 'Epoch%d.tar' % (trained_epoch+Epochs,)))
    return


def test(args):
    """
    Testing process main function

    :param args: args from command line inputs
    :return: -
    """
    # get CNN model
    model = select_model(args)
    model.cuda()

    # dataset and loader
    dataset, loader = load_dataset(args)

    # testing mode
    model.eval()
    dataset.eval()

    # load saved states
    trained_epoch, global_i = load_ckpt(args.load, model)
    save_root = os.path.dirname(os.path.dirname(args.load))
    LOG.warning("Loaded saved state at: %s" % save_root)
    LOG.warning("Trained epochs: %d" % trained_epoch)
    LOG.warning("Global steps: %d" % global_i)

    # create windows
    for i in range(BS):
        cv2.namedWindow('Prediction_%d' % i, cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Prediction_%d' % i, 800, 800)

    # start testing
    pbar = tqdm(loader)
    with torch.no_grad():
        for i, data in enumerate(pbar):
            images, targets = data
            images_torch = [torch.from_numpy(image).float().cuda() for image in images]

            # model inference
            pred = model(images_torch)

            # interpret the predictions
            interpreted_images = interpret_predictions(images, targets, pred, dataset.c2l)

            # show info
            pbar.set_description("Click any key to show next prediction....")

            for j in range(BS):
                # convert to opencv bgr for plotting
                res = cv2.cvtColor(interpreted_images[j], cv2.COLOR_RGB2BGR)
                # show image
                cv2.imshow('Prediction_%d' % j, res)

            # wait for examination until next prediction
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break

    LOG.warning("Exit. ")
    return


def interpret_predictions(images, targets, predictions, c2l, color_predict=(255, 0, 0), color_correct=(0, 255, 0)):
    """
    plot model predictions and ground truth bounding boxes on the original image

    :param images: List[ndarray], the output directly from the dataset's __getitem__, which is normalized to [0, 1],
                                  list size is the batch size, which applies to all the following params.
    :param targets: List[dict], the output directly from the dataset's __getitem__, which contains the keys: 'boxes',
                                and 'labels'.
    :param predictions: List[dict], the output directly from the model's inference, which contains the keys: 'boxes',
                                    'labels', and 'scores'
    :param c2l: dict, a dictionary convert class numbers to class labels
    :param color_predict: Tuple[int,int,int], color for the predictions
    :param color_correct: Tuple[int,int,int], color for the ground truths
    :return: List[ndarray], interpreted images in ndarray
    """
    # pred = List[dict_keys(['boxes', 'labels', 'scores']), ...]
    interpreted_images = []
    # iterating every image in the batch
    for img, target, pred_dict in zip(images, targets, predictions):
        bg = np.array(Image.fromarray((img * 255).transpose(1, 2, 0).astype(np.uint8)))  # To shape (H, W, C)
        bboxes, labels, scores = pred_dict['boxes'], pred_dict['labels'], pred_dict['scores']
        bboxes_gt, labels_gt = target['boxes'], target['labels']

        # appply nms for each class
        bboxes, scores, labels = apply_nms(bboxes, scores, labels, nmsIoU)

        # convert data
        bboxes = bboxes.detach().round().cpu().numpy().astype(np.int64)
        labels = list(map(lambda l: c2l[l], labels.detach().cpu().numpy()))
        scores = scores.detach().cpu().numpy()

        # add predicted bboxes to the image
        for k in range(bboxes.shape[0]):
            plot_one_box(bboxes[k], bg, color_predict, labels[k] + ': ' + str(np.round(scores[k] * 100, 2)))

        # add ground truth bboxes to the image
        for k in range(bboxes_gt.shape[0]):
            plot_one_box(bboxes_gt[k], bg, color_correct, c2l[labels_gt[k]], position='bottom')

        # collect plotted images
        interpreted_images.append(bg)
    return interpreted_images


def apply_nms(boxes, scores, labels, ths_IoU):
    """
    Apply non-maximum suppression on the boxes according to their intersection-over-union

    :param boxes: Tensor, shape (N, 4), boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2) format
    :param scores: Tensor, shape (N,), scores for each one of the boxes
    :param labels: Tensor, shape (N,), labels for each one of the boxes
    :param ths_IoU: float, discards all overlapping boxes with IoU > iou_threshold
    :return: filtered boxes, scores and labels
    """
    selected_bboxes, selected_labels, selected_scores = [], [], []
    for l in labels.unique():
        m = labels == l  # get label mask
        b = boxes[m]  # get corresponding boxes
        s = scores[m]  # get corresponding scores
        ll = labels[m]  # get corresponding labels
        res = nms(b, s, ths_IoU)  # indices that kept
        selected_bboxes.append(b[res])
        selected_labels.append(ll[res])
        selected_scores.append(s[res])
    if len(selected_bboxes) == 0:
        return boxes, scores, labels
    boxes = torch.cat(selected_bboxes)
    labels = torch.cat(selected_labels)
    scores = torch.cat(selected_scores)
    return boxes, scores, labels


def select_model(args):
    """
    return the corresponding model based on the command line args
    :param args: command line args
    :return: pytorch model
    """
    # use the base vgg19 with batch normalization model
    if args.resnet50:
        LOG.warning('Loading Faster R-CNN with ResNet50 backbone...')
        LOG.warning('It may take few minutes to load the PyTorch model...please wait patiently...')
        model = fasterRCNN_ResNet50_fpn()
        LOG.warning('Trainable parameters: %d' % count_parameters(model))
        return model

    # load customzied model
    elif args.simple:
        LOG.error("Customized model not implemented.")
        raise NotImplementedError("Customized model not implemented")


def load_dataset(args):
    """
    loads dataset and data loader

    :param args: command line args
    :return: Train dataset, and pytorch data loader
    """
    if args.resnet50:
        dataset = TrainCanvasDataset(data_path=args.dataset, train_len=trainLen, test_len=testLen,
                                     min_characters=minCharacters, max_characters=maxCharacters,
                                     color=True)
    else:
        dataset = TrainCanvasDataset(data_path=args.dataset, train_len=trainLen, test_len=testLen,
                                     min_characters=minCharacters, max_characters=maxCharacters)

    loader = DataLoader(dataset, batch_size=BS, shuffle=SF, num_workers=numWorkers,
                        pin_memory=pinMem, drop_last=dropLast, timeout=timeOut, collate_fn=dataset_collate_fn)
    return dataset, loader


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


def load_ckpt(path, model, optimizer=None):
    """
    Load a pre-trained model on GPU for training or evaluation

    :param model: pytorch model object to load trained parameters
    :param optimizer: optimizer object used in the last training
    :param path: path to the saved checkpoint
    """
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    global_step = ckpt['global_step']
    epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return epoch, global_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='', help="path to the saved 'tar' checkpoint file.")
    parser.add_argument('--dataset', type=str, default='./dataset/data.npz', help="Path to the dataset.")
    parser.add_argument('--train', action='store_true', help='Start training process.')
    parser.add_argument('--test', action='store_true', help='Start testing process.')
    parser.add_argument('--resnet50', action='store_true', help='Use Faster R-CNN with ResNet50 backbone.')
    parser.add_argument('--simple', action='store_true', help='Use customized Faster R-CNN.')
    opt = parser.parse_args()
    print(opt)

    if opt.train:
        train(opt)
    elif opt.test:
        test(opt)
    else:
        LOG.warning("Please specify whether to train or test")
        raise ValueError("Please specify whether to train or test using --train or --test")
