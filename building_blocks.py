import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

v_utils.save_image()

def conv_block(in_dim, out_dim, act_fn):
    """
    convolutional layer
    :param in_dim: number of input dimension or number of input filters
    :param out_dim: number of output filters
    :param act_fn: activation function
    :return: neural network model
    """
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model


def conv_trans_blok(in_dim, out_dim, act_fn):
    """
    transposed convolutional layer building block
    e.g. up-conv 2x2 in the paper
    :param in_dim: number of input dimension or number of input filters
    :param out_dim: number of output filters
    :param act_fn:
    :return: neural network model
    """
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model


def maxpool():
    """
    Applies a 2D max pooling over an input
    e.g. max pool 2x2 in the paper
    :return: maxpooling class which is able to utilize for layer after MaxPool initializing
    """
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim, out_dim, act_fn):
    """
    corresponding to first convolutional process
    e.g. input->1->64->64 in the paper
    :param in_dim: input dimensions or number of input filters)
    :param out_dim: number of filters
    :param act_fn: activation function
    :return: neural network model
    """
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


# conv 1x1 in the paper
def conv_block_3(in_dim, out_dim, act_fn):
    """
    convolutional building blocks in the paper
    :param in_dim:
    :param out_dim:
    :param act_fn:
    :return:
    """
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim)
    )
    return model