#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/01/04
"""
import torch
import torchvision


def fasterRCNN(num_classes=25):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)
    return model