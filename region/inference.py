#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN 

# In[1]:


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
import pickle
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.balloon import balloon

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "/path/to/mask_rcnn_balloon.h5"  # TODO: update this path


# ## Configurations

# In[2]:


config = balloon.BalloonConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "datasets/balloon")


# In[3]:


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3
    PRE_NMS_LIMIT = 4000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_INFERENCE = 1000
    
    
    

config = InferenceConfig()
config.display()


# ## Notebook Preferences

# In[4]:


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# In[5]:


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Load Validation Dataset

# In[6]:


# Load validation dataset
dataset = balloon.BalloonDataset()
dataset.load_balloon(BALLOON_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


# ## Load Model

# In[7]:


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


# In[8]:


# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
weights_path = model.find_last()

# Load weights
weights_path= "/home/anishg/prusty/Mask_RCNN/logs/object20190207T2135/mask_rcnn_object_0050.h5"
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
#model=load_model (weights_path)


# ## Run Detection

# In[9]:
out_dic={}
all_images_test=dataset.image_ids
cnt=0


avg_pagewise=[]
avg_classwise=[]
acc_classwise=[]
avg_fIOU=0.0
avg_mAP=0.0
avg_p=0.0
avg_r=0.0
avg_pres=[]
avg_rec=[]
avg_mAP_range=[]

for ind in range(len(all_images_test)):
    cnt+=1
    image_id=all_images_test[ind]
    print(ind," : ",image_id)
    #image_id=34
    image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    # print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
    #                                        dataset.image_reference(image_id)))
    img_name=info['id']
    print(img_name)
    # Run object detection
    # print(image.shape)
    # print(image)
    # image=cv2.imread('efeo_010_01_03.jpg',1)
    # image,_,_,_,_=utils.resize_image(image,min_dim=256, max_dim=1024)
    #print(image.shape)
    results = model.detect([image], verbose=0)
    #print(results)
    # Display results
    #ax = get_ax(1)
    r = results[0]
    #print(r)
    # ccc=visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                             dataset.class_names, r['scores'], ax=ax,
    #                             title="Predictions",show_bbox=False,show_mask=True)
    # visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, 
    #                             dataset.class_names, ax=get_ax(1),
    #                             show_bbox=False, show_mask=False,
    #                             title="Ground Truth")
    # visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, 
    #                             dataset.class_names, ax=get_ax(1),
    #                             show_bbox=False, show_mask=True,
    #                             title="Ground Truth")

    # print("gt_bbox : ",gt_bbox.shape)
    # print("gt_class_id : ",gt_class_id)
    # print("gt_mask : ",gt_mask.shape)
    # print("scores: ",r['scores'].shape )
    # print("gt_bbox : ",r['rois'].shape)
    # print("gt_class_id : ",r['class_ids'])
    # print("gt_mask : ",r['masks'].shape)
    pagewise,weighted,com_freq,mAP_out,pres_out,rec_out,mAP_range,class_wise,class_acc=utils.compute_per_region_ap(gt_bbox, gt_class_id, gt_mask,
                            r['rois'], r['class_ids'], r['scores'], r['masks'],iou_threshold=0.0,score_threshold=0.0)
    # # res1,_,_,_=utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
    # #                        r['rois'], r['class_ids'], r['scores'], r['masks'])
    # # print(res)
    # # print(res1)
    # visualize.display_differences(
    #     image,
    #     gt_bbox, gt_class_id, gt_mask,
    #     r['rois'], r['class_ids'], r['scores'], r['masks'],
    #     dataset.class_names, ax=get_ax(),
    #     show_box=False, show_mask=False,
    #     iou_threshold=0.5, score_threshold=0.5)
    avg_classwise.append(list(class_wise.values()))
    acc_classwise.append(list(class_acc.values()))
    #print(avg_classwise)
    tt_dic={}
    pres_out=np.array(pres_out)
    rec_out=np.array(rec_out)   
    p=pres_out.mean()
    r=rec_out.mean()
    print("pagewise : ",pagewise)
    print("f-weighted :",weighted)
    #print(com_freq)
    print("mean AP : ",mAP_out)
    print("mean precision : ",p)
    print("mean recall : ",r)
    print("prec_range : ",pres_out)
    print("rec_range : ",rec_out)
    print("AP_range : ",mAP_range)
    avg_pagewise.append(list(pagewise.values()))
    avg_fIOU+=weighted
    avg_mAP+=mAP_out
    avg_p+=p
    avg_r+=r
    avg_pres.append(pres_out)
    avg_rec.append(rec_out)
    avg_mAP_range.append(mAP_range)



    tt_dic['pagewise']=pagewise
    tt_dic['f-weighted']=weighted
    tt_dic['mAP']=mAP_out
    tt_dic['mprec']=p
    tt_dic['mrec']=r
    tt_dic['Precision']=pres_out
    tt_dic['recall']=rec_out
    tt_dic['maP_range']=mAP_range

    # print(res[2][1])
    # print(r['masks'].shape,gt_mask.shape)
    # res2=utils.compute_overlaps_masks(gt_mask,r['masks'])
    # print(np.array(res2).shape)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)
    out_dic[img_name]=tt_dic

    # In[ ]:
dic_freq_bhoomi={1:1.0,2:46.0,3:159.0,4:305.0,5:28.0,6:26.0,7:1.0,8:1.0,9:14.0,10:14.0}
dic_freq_PIH={1:1.0,2:2.0,3:372.0,4:6.0,5:53.0,6:73.0,7:7.0,8:13.0,9:8.0,10:52.0}

dic_freq_all={1:1.0,2:48.0,3:531,4:311,5:81,6:99,7:8.0,8:14.0,9:22.0,10:66.0}
# arr_freq=list(dic_freq.values())

avg_classwise=np.sum(avg_classwise,axis=0)
acc_classwise=np.sum(acc_classwise,axis=0)
avg_fIOU=avg_fIOU*1.0/cnt
avg_mAP=avg_mAP*1.0/cnt
avg_p=avg_p*1.0/cnt
avg_r=avg_r*1.0/cnt
avg_pres=np.mean(np.array(avg_pres),axis=0)
avg_rec=np.mean(np.array(avg_rec),axis=0)
avg_pagewise=np.mean(np.array(avg_pagewise),axis=0)
avg_mAP_range=np.mean(np.array(avg_mAP_range),axis=0)

class_weighted=0.0
for i in range(len(avg_classwise)):
    avg_classwise[i]=(avg_classwise[i]*1.0)/dic_freq_PIH[i+1]
    acc_classwise[i]=(acc_classwise[i]*1.0)/dic_freq_PIH[i+1]


print("final results: /////########################")

print("avg_IOU_classwise : ",avg_classwise)
print("acc_classwise : ",acc_classwise)
print("avg_fIOU : ",avg_fIOU)
print("avg_pagewise : ",avg_pagewise)
print("avg_mAP : ",avg_mAP)
print("avg prec : ",avg_p)
print("avg rec : ",avg_r)
print("avg_mAP_range : ",avg_mAP_range)
print("avg_pres_range : ",avg_pres)
print("avg_rec_range : ",avg_rec)


with open('pih_metrics.pickle','wb') as f:
    pickle.dump(out_dic,f) 



