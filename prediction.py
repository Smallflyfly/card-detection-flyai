# -*- coding: utf-8 -*
from flyai.framework import FlyAI
import os

from torch.utils.data import DataLoader

from dataset import CarDataset
from model.faster_rcnn import fasterRCNN
from path import MODEL_PATH
from utils.utils import load_pretrained_weights


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        model = fasterRCNN(num_classes=25)
        load_pretrained_weights(model, 'last.pt')
        return model

    def predict(self, image_path):
        '''
        模型预测返回结果
        :参数示例 image_path='./data/input/image/0.jpg'
        :return: 返回预测结果格式具体说明如下：
        '''
        # 0.jpg
        image_name = os.path.basename(image_path)
        model = self.load_model()
        model = model.cuda()
        model.eval()
        dataset = CarDataset()
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for index, data in enumerate(test_dataloader):
            im, cls_label, gt_bbox = data
            im = im.cuda()
            out = model(im, [])
            print(out)
            print(cls_label)
            print(gt_bbox)
            fang[-1]

        # 返回bbox格式为 [xmin, ymin, width, height]
        pred_result = [{"image_name": image_name, "label_name": 'I', "bbox": [735, 923, 35, 75], "confidence": 0.2},
                        {"image_name": image_name, "label_name": 'I', "bbox": [525, 535, 53, 54], "confidence": 0.3}]

        return pred_result


if __name__ == '__main__':
    prediction = Prediction()
