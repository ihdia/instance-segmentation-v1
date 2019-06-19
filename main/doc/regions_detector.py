import os
from copy import deepcopy

import cv2
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
from mrcnn import model as modellib, utils, visualize

from . import train

(dir_path, fname) = os.path.split(__file__)
PACKAGE_ROOT_DIR = os.path.normpath(
    os.path.abspath(os.path.join(dir_path, '../../')))
DEFAULT_WEIGHTS_PATH = os.path.join(PACKAGE_ROOT_DIR, 'pretrained_model_indiscapes.h5')
DEFAULT_DATASETS_DIR = os.path.join(PACKAGE_ROOT_DIR, 'datasets/doc')
DEFAULT_MODEL_DIR = os.path.join(PACKAGE_ROOT_DIR, 'logs')


response_template = {
  "_via_settings": {
    "ui": {
      "annotation_editor_height": 30,
      "annotation_editor_fontsize": 0.6000000000000001,
      "leftsidebar_width": 18,
      "image_grid": {
        "img_height": 80,
        "rshape_fill": "none",
        "rshape_fill_opacity": 0.3,
        "rshape_stroke": "yellow",
        "rshape_stroke_width": 2,
        "show_region_shape": True,
        "show_image_policy": "all"
      },
      "image": {
        "region_label": "region_id",
        "region_label_font": "10px Sans"
      }
    },
    "core": {
      "buffer_size": 18,
      "filepath": {},
      "default_filepath": ""
    },
    "project": {
      "name": "corrected_3"
    }
  },
  "_via_img_metadata": {
    "": {
      "filename": None,
      "size": -1,
      "regions": [],
      "file_attributes": {}
    }
  },
  "_via_attributes": {
    "region": {
      "Spatial Annotation": {
        "type": "dropdown",
        "description": "",
        "options": {
          "Hole(Virtual)": "",
          "Hole(Physical)": "",
          "Character Line Segment": "",
          "Boundary Line": "",
          "Physical Degradation": "",
          "Page Boundary": "",
          "Character Component": "",
          "Picture": "",
          "Decorator": "",
          "Library Marker": ""
        },
        "default_options": {}
      },
      "Comments": {
        "type": "text",
        "description": "",
        "default_value": ""
      }
    },
    "file": {}
  }
}

class DefaultInferenceConfig(train.Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    DETECTION_MIN_CONFIDENCE = 0.6
    DETECTION_NMS_THRESHOLD = 0.3
    PRE_NMS_LIMIT = 12000
    RPN_ANCHOR_SCALES = (8, 32, 64, 256, 1024)
    RPN_ANCHOR_RATIOS = [1, 3, 10]

    POST_NMS_ROIS_INFERENCE = 12000


class RegionsDetector(object):

    def __init__(
            self, weights_path=DEFAULT_WEIGHTS_PATH, dataset_dir=DEFAULT_DATASETS_DIR,
            dataset_subset='val', model_dir=DEFAULT_MODEL_DIR, config=DefaultInferenceConfig()):
        self.weights_path = weights_path
        self.dataset_dir = dataset_dir
        self.dataset_subset = dataset_subset
        self.model_dir = model_dir
        self.config = config

    def load_model(self):
        DEVICE = "/cpu:0"
        self.dataset = train.Dataset()
        self.dataset.load_data(self.dataset_dir, "val")
        self.dataset.prepare()

        with tf.device(DEVICE):
            self.model = modellib.MaskRCNN(
                mode="inference", model_dir=self.model_dir, config=self.config)

        self.model.load_weights(self.weights_path, by_name=True)
        self.graph = tf.get_default_graph()

    def detect(self, cv2img, interactive=False, should_scale=False):
        with self.graph.as_default():
            image=cv2.cvtColor(cv2img,cv2.COLOR_BGR2RGB)

            if should_scale:
                image, _, scale, padding, _ = utils.resize_image(image, min_dim=256, max_dim=1024)
            else:
                image, _, scale, padding, _ = image, None, 1, None, None

            if interactive:
                _, ax = plt.subplots(1, 1, figsize=(16*1, 16*1))
            else:
                ax = None

            results = self.model.detect([image], verbose=1)
            r = results[0]
            ccc, contours = visualize.display_instances(
                image, r['rois'], r['masks'],
                r['class_ids'], self.dataset.class_names, r['scores'],
                ax=ax, title='Predictions', show_bbox=False, show_mask=True
            )

            cls = r['class_ids']
            classes = [
                'Background', 'Hole(Virtual)', 'Hole(Physical)', 'Character Line Segment',
                'Physical Degradation', 'Page Boundary', 'Character Component', 'Picture',
                'Decorator', 'Library Marker', 'Boundary Line'
            ]

            regions = []
            for i in range(len(cls)):
                if not len(contours[i]):
                    continue
                ln = len(contours[i][0])
                all_points_x = []
                all_points_y = []
                for j in range(ln):
                    if j%20 == 0:
                        all_points_x.append((contours[i][0][j][0] - (padding[0][0] if should_scale else 0)))
                        all_points_y.append((contours[i][0][j][1] - (padding[1][0] if should_scale else 0)))

                rg = {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": all_points_x,
                        "all_points_y": all_points_y
                    },
                    "region_attributes": {
                        "Comments": "",
                        "Spatial Annotation": str(classes[cls[i]])
                    },
                    "timestamp": {
                        "StartingTime": 6016533,
                        "EndTime": 6035060
                    }
                }
                regions.append(rg)

            response = deepcopy(response_template)
            response['_via_img_metadata']['']["regions"] = regions

            return response

    def detect_from_image_file(self, img_file_path):
        img = cv2.imread(img_file_path, 1)
        return self.detect(img)

    def detect_from_image_content(self, img_content):
        np_arr = numpy.fromstring(img_content, numpy.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return self.detect(img_cv)
