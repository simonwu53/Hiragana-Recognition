import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


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
