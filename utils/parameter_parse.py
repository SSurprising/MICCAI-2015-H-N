import numpy
import os
import _net
from loss import *
from utils.tools import *


def get_network(network, num_organ, train):
    if network == 'BLSC':
        from _net.BLSC import Net
    elif network == 'UNet':
        from _net.UNet import Net
    elif network == 'VNet':
        from _net.VNet import Net
    else:
        raise('No such network', network)

    if train:
        net = Net(training=True, num_organ=num_organ)
        net.weight_init()
    else:
        net = Net(training=False, num_organ=num_organ)

    return net


def get_loss(loss_name, num_organ):
    if loss_name == 'dice':
        loss_func = DiceLoss(num_organ)
    elif loss_name == 'ce':
        loss_func = CELoss(num_organ)
    else:
        raise('No such loss', loss_name)

    return loss_func

def get_log_name(save_path, train, ensemble=None):

    if ensemble:
        suffixs = '_ensemble'
    else:
        suffixs = ''

    if train:
        log_name = join(save_path, 'train' + suffixs + '.log')
    else:
        log_name = join(save_path, 'test' + suffixs + '.log')

    return log_name
