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

from path import DATA_PATH


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
        self.transforms = transforms.Compose([transforms.ToTensor()])

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
        # resize --> 800 * 1280
        print(image)
        print(w, h)
        fang[-1]
        im = self.transforms(im)
        return im, class_label, bbox_label

    def __len__(self):
        return len(self.train_images)