#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/01/04
"""
import pickle
from collections import OrderedDict
from functools import partial

import torch


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