import cv2
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model
import cv2
import keras.backend as K
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from main.doc import train

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


config = train.BalloonConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "datasets/doc")

class InferenceConfig(config.__class__):
# Run detection on one image at a time
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	IMAGE_RESIZE_MODE = "square"
	DETECTION_MIN_CONFIDENCE = 0.6
	DETECTION_NMS_THRESHOLD = 0.3
	PRE_NMS_LIMIT = 12000
	RPN_ANCHOR_SCALES = (8,32,64,256,1024)
	RPN_ANCHOR_RATIOS = [1,3,10]

	POST_NMS_ROIS_INFERENCE = 12000
	
	'''
	
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	IMAGE_RESIZE_MODE = "square"
	DETECTION_MIN_CONFIDENCE = 0.3
	DETECTION_NMS_THRESHOLD = 0.3
	PRE_NMS_LIMIT = 12000
	RPN_ANCHOR_SCALES = (8,32,64,256,1024)
	RPN_ANCHOR_RATIOS = [1,3,10]

	POST_NMS_ROIS_INFERENCE = 12000
	'''
def get_ax(rows=1, cols=1, size=16):
	_, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
	return ax
def runtest(img):

	K.clear_session()

	config = InferenceConfig()
	DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

	# Inspect the model in training or inference modes
	# values: 'inference' or 'training'
	# TODO: code for 'training' test mode not ready yet
	TEST_MODE = "inference"

	dataset = train.BalloonDataset()
	dataset.load_balloon(BALLOON_DIR, "val")

	# Must call before using the dataset
	dataset.prepare()

	print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
	with tf.device(DEVICE):
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
							  config=config)
	#weights_path = model.find_last()

	# Load weights
	weights_path= "/home/abhishek/prusty/Mask_RCNN/logs/object20190207T2135/mask_rcnn_object_0054.h5"

	print("Loading weights ", weights_path)

	model.load_weights(weights_path, by_name=True)
	#model=load_model (weights_path)
	import json
	image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	image,_,scale,padding,_=utils.resize_image(image,min_dim=256, max_dim=1024)
	print(padding)
	# plt.figure(figsize=(15,15))
	# plt.axis('off')
	# plt.imshow(image)
	print(image.shape)
	results = model.detect([image], verbose=1)
	#print(results)
	# Display results
	ax = get_ax(1)
	r = results[0]
	#print(r)
	ccc,contours=visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
	                            dataset.class_names, r['scores'], ax=ax,
	                            title="Predictions",show_bbox=False,show_mask=True)

	print(contours[0][0])
	print(contours[0])
	cls=r['class_ids']
	classes = ['Background','Hole(Virtual)','Hole(Physical)','Character Line Segment',
	           'Physical Degradation','Page Boundary','Character Component','Picture',
	           'Decorator','Library Marker','Boundary Line']

	strt="""
	{
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
	        "show_region_shape": true,
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
	      "filename": \""""+str("/../../prusty/Mask_RCNN/samples/balloon/static/images/1.jpg")+"""\",
	      "size": -1,
	      "regions": [
	"""

	end="""
	],
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
	"""

	rgns=""
	for i in range(len(cls)):
	    str1=""
	    str2=""
	    ln=len(contours[i][0])
	    print(ln)
	    for j in range(ln):
	        if(j%20==0):
	            str1+=str(contours[i][0][j][0]-padding[0][0])
	            str1+=','
	            str1+='\n'
	    for j in range(ln):
	        if(j%20==0):
	            str2+=str(contours[i][0][j][1]-padding[1][0])
	            str2+=','
	            str2+='\n'
	    str1=str1[:-2]
	    str2=str2[:-2]
	    rg="""{
	          "shape_attributes": {
	            "name": "polygon",
	            "all_points_x": [ """+ str2+"""],
	            "all_points_y": ["""+str1+ """]
	          },
	          "region_attributes": {
	            "Spatial Annotation":\""""+str(classes[cls[i]])+"""\",
	            "Comments": ""
	          },
	          "timestamp": {
	            "StartingTime": 6016533,
	            "EndTime": 6035060
	          }
	        }"""
	    if(i!=len(cls)-1):
	        rg+=","
	    rgns+=rg

	with open ('save.json','w') as f:
	    f.write(strt)
	    f.write(rgns)
	    f.write(end)
	h, w = image.shape[:2]
	image=image[padding[0][0]:h-padding[0][1],padding[1][0]:w-padding[1][1]]
	print(scale)
	# image = utils.resize(image, (round(h *scale), round(w *scale)),
	#                        preserve_range=True)
	# cv2.imwrite('/../../prusty/Mask_RCNN/samples/balloon/static/images/1.jpg', cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2BGR))
	plt.savefig('/home/abhishek/prusty/Instance-segmentation/main/doc/static/images/2.jpg',bbox_inches='tight')
