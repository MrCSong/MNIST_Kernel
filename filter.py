import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import math


def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def plot_conv_weights(weights, plot_dir, name, channels_all=True, filters_all=True, channels=[0], filters=[0]):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """

    w_min = np.min(weights)
    w_max = np.max(weights)

    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    if filters_all:
        num_filters = weights.shape[3]
        filters = range(weights.shape[3])
    else:
        num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel_ID in channels:
        # iterate filters inside every channel
        if num_filters == 1:
            img = weights[:, :, channel_ID, filters[0]]
            axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            for l, ax in enumerate(axes.flat):
                # get a single filter
                img = weights[:, :, channel_ID, filters[l]]
                # put it on the grid
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel_ID)), bbox_inches='tight')


plot_conv_weights("D:\\PyCharm\\Project\\VTAB\\Weight\\MNIST.h5","D:\\PyCharm\\Project\\VTAB\\Weight\\",'Filter')