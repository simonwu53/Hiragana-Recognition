import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import logging
import cv2


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Visualization')


def grid_plot(img_list, nrows=5, labels=None, cmap='gray', show=True):
    """
    show images in grid view

    :param img_list: List[ndarray,...], a list of images for plotting, each has shape (50, 50)
    :param nrows: int, number of plots per row, default=5
    :param labels: Optional[List[str,...]], None or a list of labels that has the same length as img_list,
                                            each label is a string.
    :param cmap: str, plotting color map, default 'gray'
    :param show: bool, whether to show the grid plots
    :return: -
    """
    rows = len(img_list) // nrows + 1
    fig = plt.figure(figsize=(20, 25))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, nrows),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for i, ax in enumerate(grid):
        ax.imshow(img_list[i], cmap=cmap)
        ax.set_title(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])

        if i == len(img_list) - 1:
            break

    if show:
        plt.show()
    return


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
