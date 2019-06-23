#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
from doc import train


# In[11]:


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Path to load pretrained weights on Indiscapes
weights_path = "../../pretrained_model_indiscapes.h5" # TODO: update this path


# In[12]:


config = train.Config()
DOC_DIR = os.path.join(ROOT_DIR, "datasets/doc/")


# In[14]:


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


# In[15]:


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# In[16]:


# Load validation dataset
dataset = train.Dataset()
dataset.load_data(DOC_DIR, "test")
# Must call before using the dataset
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# In[8]:


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


# In[18]:


# Set path to balloon weights file

# Download file from the Releases page and set its path
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
# weights_path = model.find_last()
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


# In[ ]:


out_dic={}
all_images_test=dataset.image_ids
cnt=0
avg_fIOU=0.0
avg_mAP=0.0
avg_p=0.0
avg_r=0.0
avg_pres=[]
avg_rec=[]
instance_count=[0]*10
avg_iou_classwise=[]
avg_acc_classwise=[]
avg_mAP_range=[]
for ind in range(len(all_images_test)):
    cnt+=1
    image_id=all_images_test[ind]
    print(ind," : ",image_id)
    image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    # print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
    #                                        dataset.image_reference(image_id)))
    unique_ids=np.unique(gt_class_id)
    for i in unique_ids:
        instance_count[i-1]+=1
    print(unique_ids)
    img_name=info['id']
    print(img_name)
    results = model.detect([image], verbose=0)
    r = results[0]
    #TODO:Change the iou_threshold to measure mAP 
    pagewise_iou,weighted,com_freq,mAP_out,pres_out,rec_out,mAP_range,class_wise,pagewise_acc=utils.compute_per_region_ap(gt_bbox, gt_class_id, gt_mask,
                        r['rois'], r['class_ids'], r['scores'], r['masks'],iou_threshold=0.5,score_threshold=0.5)
    avg_iou_classwise.append(list(pagewise_iou.values()))
    avg_acc_classwise.append(list(pagewise_acc.values()))
    pres_out=np.array(pres_out)
    rec_out=np.array(rec_out)
    p=pres_out.mean()
    r=rec_out.mean()
    avg_fIOU+=weighted
    avg_mAP+=mAP_out
    avg_p+=p
    avg_r+=r
    avg_pres.append(pres_out)
    avg_rec.append(rec_out)
    avg_mAP_range.append(mAP_range)
    tt_dic={}
    tt_dic['pagewise_iou']=pagewise_iou
    tt_dic['pagewise_acc']=pagewise_acc
    tt_dic['mAP']=mAP_out
    tt_dic['mprec']=p
    tt_dic['mrec']=r
    tt_dic['Precision']=pres_out
    tt_dic['recall']=rec_out
    tt_dic['maP_range']=mAP_range
    out_dic[img_name]=tt_dic
        
avg_fIOU=avg_fIOU*1.0/cnt
avg_mAP=avg_mAP*1.0/cnt
avg_p=avg_p*1.0/cnt
avg_r=avg_r*1.0/cnt
avg_pres=np.mean(np.array(avg_pres),axis=0)
avg_rec=np.mean(np.array(avg_rec),axis=0)
avg_mAP_range=np.mean(np.array(avg_mAP_range),axis=0)
avg_mAP_sum=np.mean(avg_mAP_range)
# print("final results: /////########################")
avg_iou_pgwise_sum=list(np.sum(avg_iou_classwise,axis=0))
avg_acc_pgwise_sum=list(np.sum(avg_acc_classwise,axis=0))
Final_classwise_iou= [x/y if y else 0 for x,y in zip(avg_iou_pgwise_sum,instance_count)]
Final_classwise_acc= [w/z if z else 0 for w,z in zip(avg_acc_pgwise_sum,instance_count)]
classes = ['Hole(Virtual)','Hole(Physical)','Character Line Segment','Physical Degradation','Page Boundary','Character Component','Picture','Decorator','Library Marker','Boundary Line']
class_iou=list(zip(classes,Final_classwise_iou))
class_acc=list(zip(classes,Final_classwise_acc))
class_count=list(zip(classes,instance_count))
print("avg_IOU_classwise : ",class_iou)
print("avg_acc_classwise : ",class_acc)
print("class_count : ",class_count)
print("avg_fIOU : ",avg_fIOU)
print("avg_mAP : ",avg_mAP)
# print("avg prec : ",avg_p)
# print("avg rec : ",avg_r)
print("avg_mAP_range_sum : ",avg_mAP_sum)
#print("avg_mAP_range : ",avg_mAP_range)
# print("avg_pres_range : ",avg_pres)
# print("avg_rec_range : ",avg_rec)

with open('PIH_metrics.pickle','wb') as f:
    pickle.dump(out_dic,f) 


# In[ ]:




