#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/01/04
"""
import os

import pandas
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from path import DATA_PATH
from utils.utils import show_image


class CarDataset(Dataset):
    def __init__(self):
        super(CarDataset, self).__init__()
        self.card_root = os.path.join(DATA_PATH, 'CardDetection')
        self.image_path = os.path.join(self.card_root, 'images')
        self.train_csv = os.path.join(self.card_root, 'train.csv')
        self.train_images = []
        self.train_classes_labels = []
        self.train_bbox_labels = []
        self.train_labels = []
        self.greek_nums_map = {}
        self.train_image_label_map = {}
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
        ])
        self.image_size = 640


        self._generate_greek_nums_map()
        self._process_data()

    def _generate_greek_nums_map(self):
        greek_nums = [
            'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
            'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
            'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX'
        ]
        for i, num in enumerate(greek_nums):
            self.greek_nums_map[num] = i

    def _process_data(self):
        images = os.listdir(self.image_path)
        self._read_csv()
        for image in images:
            self.train_images.append(os.path.join(self.image_path, image))
            all_labels = self.train_image_label_map[image]
            nums_labels = []
            bbox_labels = []
            for label in all_labels:
                nums_labels.append(label[0])
                bbox_labels.append(label[1:])
            self.train_classes_labels.append(nums_labels)
            self.train_bbox_labels.append(bbox_labels)

    def _read_csv(self):
        dataframe = pandas.read_csv(self.train_csv)
        image_paths = dataframe['image_path']
        label_greeks = dataframe['label']
        xmins, ymins, xmaxs, ymaxs = dataframe['xmin'], dataframe['ymin'], dataframe['xmax'], dataframe['ymax']
        for i in range(image_paths.shape[0]):
            image = image_paths[i]
            image = image.replace('images/', '')
            label = self.greek_nums_map[label_greeks[i]]
            xmin, ymin, xmax, ymax = xmins[i], ymins[i], xmaxs[i], ymaxs[i]
            if image not in self.train_image_label_map:
                self.train_image_label_map[image] = []
            self.train_image_label_map[image].append([label, xmin, ymin, xmax, ymax])

    def __getitem__(self, index):
        image = self.train_images[index]
        class_label = self.train_classes_labels[index]
        bbox_label = self.train_bbox_labels[index]
        im = Image.open(image)
        w, h = im.size
        im_size_max = max(w, h)
        im_scale = self.image_size / im_size_max
        new_w = int(w * im_scale)
        new_h = int(h * im_scale)
        new_im = np.zeros((640, 640, 3)).astype(int)
        im_resize = im.resize((new_w, new_h), Image.ANTIALIAS)
        # print(new_w, new_h)
        assert new_w != 640 or new_h != 640
        new_im[:new_h, :new_w, :] = np.asarray(im_resize)[:, :, :]
        new_im = Image.fromarray(np.uint8(new_im))
        new_bboxes = []
        for i in range(len(bbox_label)):
            bboxes = [bb * im_scale for bb in bbox_label[i]]
            new_bboxes.append(bboxes)
        # print(image, im_scale)
        # show_image(new_im, new_bboxes)
        # new_im.show()
        # fang[-1]
        im = self.transforms(new_im)
        # print(im.size())
        return im, class_label, new_bboxes

    def __len__(self):
        return len(self.train_images)