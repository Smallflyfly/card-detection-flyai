# -*- coding: utf-8 -*
from PIL import Image
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
        model = fasterRCNN(num_classes=25, box_score_thresh=0.05)
        load_pretrained_weights(model, './last.pth')
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
        # test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        pred_result = []
        # for index, data in enumerate(test_dataloader):
        #     im, cls_label, gt_bbox = data
        #     im = im.cuda()
        #     out = model(im, None)[0]
        #     bboxes_pred = out['boxes']
        #     labels_pred = out['labels']
        #     scores_pred = out['scores']
        #     bboxes_pred = bboxes_pred.tolist()
        #     # bboxes_pred = [bbox.tolist() for bbox in bboxes_pred]
        #     for index, bbox in enumerate(bboxes_pred):
        #         d = {}
        #         d["image_name"] = image_name
        #         label = labels_pred[index].cpu().detach().numpy()
        #         # print(label)
        #         # print(dataset.greek_nums_index_map)
        #         label_name = dataset.greek_nums_index_map[str(label)]
        #         d['label_name'] = label_name
        #         xmin, ymin, xmax, ymax = bbox[:]
        #         d['bbox'] = [int(xmin), int(ymin), int(xmax-xmin+1), int(ymax-ymin+1)]
        #         score = scores_pred[index].cpu().detach().numpy()
        #         d['confidence'] = score
        #         pred_result.append(d)
        im = Image.open(image_path)
        im = dataset.transforms(im)
        im = im.unsqueeze(0)
        im = im.cuda()

        out = model(im, None)[0]
        bboxes_pred = out['boxes']
        labels_pred = out['labels']
        scores_pred = out['scores']
        bboxes_pred = bboxes_pred.tolist()
        for index, bbox in enumerate(bboxes_pred):
            d = {}
            d["image_name"] = image_name
            label = labels_pred[index].cpu().detach().numpy()
            label_name = dataset.greek_nums_index_map[str(label)]
            d['label_name'] = label_name
            xmin, ymin, xmax, ymax = bbox[:]
            d['bbox'] = [int(xmin), int(ymin), int(xmax-xmin+1), int(ymax-ymin+1)]
            score = scores_pred[index].cpu().detach().numpy().tolist()
            d['confidence'] = score
            pred_result.append(d)

        # 返回bbox格式为 [xmin, ymin, width, height]
        # pred_result = [{"image_name": image_name, "label_name": 'I', "bbox": [735, 923, 35, 75], "confidence": 0.2},
        #                 {"image_name": image_name, "label_name": 'I', "bbox": [525, 535, 53, 54], "confidence": 0.3}]
        # print(pred_result)

        return pred_result


if __name__ == '__main__':
    prediction = Prediction()
    result = prediction.predict('data/input/CardDetection/images/5.jpg')
    print(result)
