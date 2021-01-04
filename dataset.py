#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/01/04
"""
import os

import pandas
from torch.utils.data import Dataset
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
        self.greek_nums_map = {}
        self.train_image_label_map = {}

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

    def _read_csv(self):
        dataframe = pandas.read_csv(self.train_csv)
        image_path = dataframe['image_path']
        label_greek = dataframe['label']
        xmin, ymin, xmax, ymax = dataframe['xmin'], dataframe['ymin'], dataframe['xmax'], dataframe['ymax']
        for i in range(image_path.shape[0]):
            image = image_path[i]


    def __getitem__(self, item):
        pass

    def __len__(self):
        pass