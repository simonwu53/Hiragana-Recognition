import numpy as np
import torch
from tqdm import tqdm


def train_dataset_statistics(dataset, img_shape=(50, 50)):
    # create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                         pin_memory=True, drop_last=False)

    # statistics
    n_samples = len(dataset)
    n_pixels_per_frame = np.prod(img_shape)
    x_sum = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # cam depth sum of x
    x2_sum = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # cam depth sum of x^2
    n_pixels_total = torch.tensor(n_samples*n_pixels_per_frame, dtype=torch.float32,
                                  device='cuda', requires_grad=False)  # total pixels

    # iterating over dataset
    for i, data in enumerate(tqdm(loader)):
        # unpack data
        image, label = data[0].cuda(), data[1].cuda()

        # track the sum of x and the sum of x^2
        with torch.no_grad():
            x_sum += torch.sum(image[0])
            x2_sum += torch.sum(image[0]**2)

    # calculate mean and std.
    # formula: stddev = sqrt((SUM[x^2] - SUM[x]^2 / n) / (n-1))
    mean = x_sum / n_pixels_total
    std = torch.sqrt((x2_sum-x_sum**2/n_pixels_total)/(n_pixels_total-1))

    # print results
    print('Training data Statistics: ')
    print('Mean: %.4f' % mean.cpu().item())
    print('STD: %.4f' % std.cpu().item())

    return mean, std  # calculated results: mean=23.8014, std=45.9789
