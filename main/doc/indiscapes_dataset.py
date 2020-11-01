import json
import os

import cv2
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

images_root = 'images'  # This must contain 2 folders named Bhoomi_data and penn_in_hand
via_json_root = (
    'doc'  # This must contain 3 folders named train, test, val; each with a via_region_data.json
)

categories_dict = {
    'Hole(Virtual)': 0,
    'Hole(Physical)': 1,
    'Character Line Segment': 2,
    'Boundary Line': 3,
    'Physical Degradation': 4,
    'Page Boundary': 5,
    'Character Component': 6,
    'Picture': 7,
    'Decorator': 8,
    'Library Marker': 9,
}

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


def get_indiscapes_dicts(img_dir, doc_dir):
    json_file = os.path.join(doc_dir, 'via_region_data.json')
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns['_via_img_metadata'].values()):
        record = {}
        url = v['filename']
        filename = '/'.join(url.split('/')[3:]).replace('%20', ' ')
        filename = os.path.join(img_dir, filename)
        height, width = cv2.imread(filename).shape[:2]
        record['file_name'] = filename
        record['height'] = height
        record['width'] = width
        record['image_id'] = idx

        annos = v['regions']
        objs = []
        for idx, anno in enumerate(annos):
            shape = anno['shape_attributes']
            try:
                px = shape['all_points_x']
                py = shape['all_points_y']

                if len(px) < 6:
                    # print(record, idx, len(shape["all_points_x"]))
                    while len(px) < 6:
                        px.insert(1, (px[0] + px[1]) / 2)
                        py.insert(1, (py[0] + py[1]) / 2)
                    # print(record, idx, len(shape["all_points_x"]))

                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                region = anno['region_attributes']['Spatial Annotation']

                obj = {
                    'bbox': [np.min(px), np.min(py), np.max(px), np.max(py)],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [poly],
                    'category_id': categories_dict[region],
                }
                objs.append(obj)
            except KeyError:  # Rectanges
                #                 try:
                #                     px = shape[""]

                #                 except:
                print(record, idx)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_dataset(combined_train_val=False):
    DatasetCatalog.clear()
    for d in (
        ['train_val_combined', 'val', 'test'] if combined_train_val else ['train', 'val', 'test']
    ):
        DatasetCatalog.register(
            'indiscapes_' + d,
            lambda d=d: get_indiscapes_dicts(images_root, os.path.join(via_json_root, d)),
        )
        MetadataCatalog.get('indiscapes_' + d).set(thing_classes=categories_list)
        MetadataCatalog.get('indiscapes_' + d).set(evaluator_type='indiscapes')
