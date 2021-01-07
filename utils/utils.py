#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/01/04
"""
import pickle
from collections import OrderedDict
from functools import partial

import cv2
import torch
import numpy as np


def load_checkpoint(weight_path):
    if weight_path is None:
        raise RuntimeError("模型不存在")
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(weight_path, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(weight_path, pickle_module=pickle, map_location=map_location)
    except Exception:
        raise RuntimeError("无法加载权值文件")
    return checkpoint


def load_pretrained_weights(model, weight_path):
    check_point = load_checkpoint(weight_path)
    if 'state_dict' in check_point:
        state_dict = check_point['state_dict']
    else:
        state_dict = check_point
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('Successfully load weights from {}'.format(weight_path))


def build_optimizer(model, optim='adam', lr=0.005, weight_decay=5e-04, momentum=0.9, sgd_dampening=0,
                    sgd_nesterov=False, rmsprop_alpha=0.99, adam_beta1=0.9, adam_beta2=0.99, staged_lr=False,
                    new_layers='', base_lr_mult=0.1):
    param_groups = model.parameters()
    optimizer = None
    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    return optimizer


def build_scheduler(optimizer, lr_scheduler='single_step', stepsize=1, gamma=0.1, max_epoch=1):
    global scheduler
    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch))

    return scheduler


def show_image(im, bboxes):
    # im PIL -> opencv
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox[:]
        print(xmin, ymin, xmax, ymax)
        cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 1)

    cv2.imshow('im', im)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
