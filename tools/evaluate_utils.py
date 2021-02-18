#!/usr/bin/env
# coding:utf-8
"""
Created on 2020/6/4 12:50

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import numpy as np
import torch

def evaluate_regression(pred, label):
    '''
    :param pred: numpy
    :param label: numpy
    :return:
    '''
    if type(pred) == torch.Tensor:
        pred = pred.detach().to('cpu').numpy()
    if type(label) == torch.Tensor:
        label = label.detach().to('cpu').numpy()

    mse = np.mean(np.square(pred - label))

    mae = np.mean(np.abs(pred - label))

    mape = np.mean(np.abs((pred - label) / label))
    return mse, mae, mape


