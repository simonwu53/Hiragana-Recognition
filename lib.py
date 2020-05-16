import numpy as np
import torch
from tqdm import tqdm
import logging
import random
import cv2


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Lib')


def train_dataset_statistics(dataset, img_shape=(50, 50, 1)):
    """
    Calculate the training data set statistics (mean and std.)
    :param dataset: dataset instance, if channels==3, dataset must can provide 3-channel images
    :param img_shape: image shape for the model input, shape (Height, Width, Channels)
    :return: (mean, std.) values
    """
    assert len(
        img_shape) == 3, "Image shape must be a tuple of three integers (H, W, C)."
    H, W, C = img_shape

    dataset.train()  # set to training mode
    # create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                         pin_memory=True, drop_last=False)

    # statistics
    n_samples = len(dataset)
    n_pixels_per_frame = H*W  # number of pixels per frame per channel
    # number of pixels of total frames per channel
    n_pixels_total = n_samples * n_pixels_per_frame
    if C == 1:
        x_sum = torch.tensor(0, dtype=torch.float32, device='cuda',
                             requires_grad=False)  # cam depth sum of x
        # cam depth sum of x^2
        x2_sum = torch.tensor(0, dtype=torch.float32,
                              device='cuda', requires_grad=False)

    elif C == 3:
        x_sum_r = torch.tensor(0, dtype=torch.float32,
                               device='cuda', requires_grad=False)
        x_sum_g = torch.tensor(0, dtype=torch.float32,
                               device='cuda', requires_grad=False)
        x_sum_b = torch.tensor(0, dtype=torch.float32,
                               device='cuda', requires_grad=False)
        x2_sum_r = torch.tensor(0, dtype=torch.float32,
                                device='cuda', requires_grad=False)
        x2_sum_g = torch.tensor(0, dtype=torch.float32,
                                device='cuda', requires_grad=False)
        x2_sum_b = torch.tensor(0, dtype=torch.float32,
                                device='cuda', requires_grad=False)

    else:
        LOG.error('Invalid channels number.')
        raise ValueError('Invalid channels number.')

    # iterating over dataset
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            # unpack data, image shape(B,C,H,W), B==1, C==1 or 3
            image, label = data[0].cuda(), data[1].cuda()

            # track the sum of x and the sum of x^2
            if C == 1:
                x_sum += torch.sum(image[0][0])
                x2_sum += torch.sum(image[0][0] ** 2)
            else:
                x_sum_r += torch.sum(image[0][0])
                x_sum_g += torch.sum(image[0][1])
                x_sum_b += torch.sum(image[0][2])
                x2_sum_r += torch.sum(image[0][0] ** 2)
                x2_sum_g += torch.sum(image[0][1] ** 2)
                x2_sum_b += torch.sum(image[0][2] ** 2)

    # calculate mean and std.
    # formula: stddev = sqrt((SUM[x^2] - SUM[x]^2 / n) / (n-1))
    if C == 1:
        mean = x_sum / n_pixels_total
        std = torch.sqrt((x2_sum-x_sum**2/n_pixels_total)/(n_pixels_total-1))
        print('Training data Statistics: ')
        print('Mean: %.4f' % mean.cpu().item())
        print('STD: %.4f' % std.cpu().item())
        return mean, std  # calculated results: mean=23.8014, std=45.9789
    else:
        mean_r = x_sum_r / n_pixels_total
        mean_g = x_sum_g / n_pixels_total
        mean_b = x_sum_b / n_pixels_total
        std_r = torch.sqrt(
            (x2_sum_r - x_sum_r ** 2 / n_pixels_total) / (n_pixels_total - 1))
        std_g = torch.sqrt(
            (x2_sum_g - x_sum_g ** 2 / n_pixels_total) / (n_pixels_total - 1))
        std_b = torch.sqrt(
            (x2_sum_b - x_sum_b ** 2 / n_pixels_total) / (n_pixels_total - 1))
        print('Training data Statistics: ')
        print('R channel Mean: %.4f' % mean_r.cpu().item())
        print('G channel Mean: %.4f' % mean_g.cpu().item())
        print('B channel Mean: %.4f' % mean_b.cpu().item())
        print('R channel STD: %.4f' % std_r.cpu().item())
        print('G channel STD: %.4f' % std_g.cpu().item())
        print('B channel STD: %.4f' % std_b.cpu().item())
        return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)


def sample_images(images, labels, n_samples=500):
    assert images.shape[0] > n_samples, 'Sample quantity must be smaller than the quantity of the dataset.'
    assert n_samples > 100, 'Too less samples.'
    idx = np.random.choice(images.shape[0], n_samples, replace=False)
    return images[idx], labels[idx]


def count_parameters(model):
    """Calculate number of total parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_random_canvas(characters):
    """
    Create canvas with input characters copied on to it

    :param characters: np.array() (n, H, W)

    :return: canvas np.array() (H*(n+1) x W*(n+1)), List of bounding boxes (x_1, y_1, x_2, y_2)
    """
    n = characters.shape[0]
    W, H = characters.shape[1:]
    canvas = np.zeros(((n+1)*H, (n+1)*W), np.uint8)

    def get_random_location(n, W, H):
        x1, y1 = random.randint(0, n*H-1), random.randint(0, n*W-1)
        x2, y2 = x1 + H, y1 + W
        return (x1, y1, x2, y2)
    locs = []
    retry = True
    while retry:
        locs = []
        for c in characters:
            locs.append(get_random_location(n, W, H))
        ok = []
        for loc1 in locs:
            for loc2 in locs:
                if loc1 != loc2:
                    if abs(loc1[0] - loc2[0]) > W or abs(loc1[1] - loc2[1]) > H:
                        ok.append(0)
                    else:
                        ok.append(1)
        retry = np.sum(ok) != 0

    for i in range(len(locs)):
        loc = locs[i]
        canvas[loc[1]:loc[3], loc[0]:loc[2]] = characters[i]
    return canvas, np.array(locs, np.int64)


def plot_one_box(x, img, color=None, label=None, line_thickness=None, position='top'):
    """
    adds one bounding box and label to the image

    :param x: List[int], Bounding box boundaries that contains four values
                         e.g. (wmin, hmin, wmax, hmax) -> Top-left corner and Bottom-right corner, W-width, H-height
    :param img: ndarray, the image will be annotated, shape (H, W, 3) or (H, W)
    :param color: List[int], A list of integers that are [R, G, B] values
    :param label: str, a string that will be put on the top of the bounding box
    :param line_thickness: int, an integer value indicates the bounding box's line width
    :param position: str, choose from 'top', 'bottom', specifying the label position
    :return: None, the input image itself will be updated (in-place).
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        if position == 'bottom':
            box_h = c2[1]-c1[1]
            c1 = c1[0], c1[1] + box_h + t_size[1] + 3
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        else:
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
