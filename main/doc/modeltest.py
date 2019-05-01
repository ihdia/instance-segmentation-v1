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
import keras.backend as K
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from main.doc import train

config = train.Config()
DOCDATA = ROOT_DIR+"datasets/doc"
OUTPUTPATH=  ROOT_DIR+"/main/doc/static/images/2.jpg"
IMG1PATH= ROOT_DIR+"/main/doc/static/images/1.jpg"

def get_ax(rows=1, cols=1, size=16):
	_, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
	return ax
def runtest(img,model,dataset):
	import json
	image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	image,_,scale,padding,_=utils.resize_image(image,min_dim=256, max_dim=1024)
	results = model.detect([image], verbose=1)
	ax = get_ax(1)
	r = results[0]
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
	      "filename": \""""+str(IMG1PATH)+"""\",
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
	plt.savefig(OUTPUTPATH,bbox_inches='tight')
