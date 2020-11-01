import json
import time

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

categories_list = [
    'Hole(Virtual)',
    'Hole(Physical)',
    'Character Line Segment',
    'Boundary Line',
    'Physical Degradation',
    'Page Boundary',
    'Character Component',
    'Picture',
    'Decorator',
    'Library Marker',
]


class IndiscapesPredictor:
    def __init__(self):
        cfg = get_cfg()

        cfg.merge_from_file("configs/mask_rcnn_R_50_FPN_1x_dconv_c3-c5_wideanc1.yaml")
        cfg.MODEL.WEIGHTS = "model_final.pth"
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img_path="test.jpg", write_to_file=True):
        try:
            im = cv2.imread(img_path)
            tick = time.time()
            outputs = self.predictor(im)
            tock = time.time()
            td = tock - tick

            all_points = torch.nonzero(outputs['instances'].pred_masks == True)
            all_points[torch.where(all_points[:, 0] == 1)][:, 1:].cpu().tolist()

            regions = list()
            for i in range(len(outputs['instances'])):
                regions.append({
                    "class_ids": [categories_list[outputs['instances'].pred_classes[i]]],
                    "flag": 2,
                    "exec_time": td,
                    "url": "test.jpg",
                    "error": 0,
                    "points": all_points[torch.where(all_points[:, 0] == i)][:, 1:].cpu().tolist()
                })
        except Exception as e:
            print(e)
            regions = {
                "flag": 2,
                "url": "test.jpg",
                "error": 1,
            }
        json_output = {
            'regions': regions
        }

        if write_to_file:
            with open('result.json', 'w') as fp:
                json.dump(json_output, fp)
        else:
            return json_output


if __name__ == '__main__':
    predictor = IndiscapesPredictor()
    predictor(img_path="3.jpg")
