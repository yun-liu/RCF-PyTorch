import os
import logging
import numpy as np
import torch
import torch.nn.functional as F


class Logger(object):
    def __init__(self, path='log.txt'):
        self.logger = logging.getLogger('Logger')
        self.file_handler = logging.FileHandler(path, 'w')
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)

    def info(self, txt):
        self.logger.info(txt)

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Cross_entropy_loss(prediction, label):
    mask = label.clone()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    # mask[mask == 2] = 0
    selected_idx = mask != 2
    prediction = prediction[selected_idx]
    label = label[selected_idx]
    mask = mask[selected_idx]
    cost = F.binary_cross_entropy(prediction, label, weight=mask, reduce=False)
    return torch.sum(cost)
