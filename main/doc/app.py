import os
from flask import Flask, render_template, request, jsonify,redirect,session,flash
from werkzeug import secure_filename
from modeltest import *
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

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


config = train.BalloonConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "datasets/doc")

app = Flask(__name__)
app.secret_key = 'f3cfe9ed8fae309f02079dbf'
UPLOAD_FOLDER = '/home/abhishek/prusty/Instance-segmentation/main/doc/static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model =None
class InferenceConfig(config.__class__):
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
def load_model():
	config = InferenceConfig()
	DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
	TEST_MODE = "inference"
	global dataset
	dataset = train.BalloonDataset()
	dataset.load_balloon(BALLOON_DIR, "val")
	dataset.prepare()

	print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
	global model
	with tf.device(DEVICE):
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
							  config=config)
	weights_path= "/home/abhishek/prusty/Mask_RCNN/logs/object20190207T2135/mask_rcnn_object_0054.h5"
	print("Loading weights ", weights_path)
	model.load_weights(weights_path, by_name=True)
	global graph
	graph = tf.get_default_graph()

# main route
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/uploader', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], "1.jpg"))
		flash('File successfully uploaded')
		return redirect('/')
		

@app.after_request
def add_header(r):

	r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
	r.headers["Pragma"] = "no-cache"
	r.headers["Expires"] = "0"
	r.headers['Cache-Control'] = 'public, max-age=0'
	return r

@app.route('/background_process_test')
def background_process_test():
	filepath="/home/abhishek/prusty/Instance-segmentation/main/doc/static/images/1.jpg"
	img=cv2.imread(filepath,1)
	with graph.as_default(): 
		runtest(img,model,dataset)
	return jsonify({"a":1,"b":2,"c":3})	

if __name__ == '__main__':
	load_model()
	app.run('0.0.0.0', debug=True)
