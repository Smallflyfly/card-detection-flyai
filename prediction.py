# -*- coding: utf-8 -*
from flyai.framework import FlyAI
import os
from path import MODEL_PATH


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        pass

    def predict(self, image_path):
        '''
        模型预测返回结果
        :参数示例 image_path='./data/input/image/0.jpg'
        :return: 返回预测结果格式具体说明如下：
        '''

        image_name = os.path.basename(image_path) # 0.jpg

        # ... 模型预测

        # 返回bbox格式为 [xmin, ymin, width, height]
        pred_result = [{"image_name": image_name, "label_name": 'I', "bbox": [735, 923, 35, 75], "confidence": 0.2},
                        {"image_name": image_name, "label_name": 'I', "bbox": [525, 535, 53, 54], "confidence": 0.3}]

        return pred_result
