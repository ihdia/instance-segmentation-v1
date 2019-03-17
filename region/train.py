"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

python3 balloon.py train --dataset=/home/sowmya.aitha/prusty/Mask_RCNN/ --weights=coco


Usage: import the module (see Jupyter notebooks for examples), or run from
	   the command line as such:

	# Train a new model starting from pre-trained COCO weights
	python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

	# Resume training a model that you had trained earlier
	python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

	# Train a new model starting from ImageNet weights
	python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

	# Apply color splash to an image
	python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

	# Apply color splash to video using the last weights you trained
	python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa
from skimage.transform import resize
import cv2
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
EPOCH=300
def make_mask_weight(masks):
    masks=np.array(masks)
    #print(masks.shape)
    masks = np.reshape(masks, (-1, masks.shape[1], masks.shape[2]))
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros_like(masks)
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = ndimage.distance_transform_edt(mask)
        #print(bounds)
        mx = np.max(bounds)
        bounds[bounds>0]=mx-bounds[bounds>0]
        #print(bounds)
        #print(bounds.shape)
        distMap[i:] = bounds

    #print(distMap)
    return distMap
class BalloonConfig(Config):
	"""Configuration for training on the toy  dataset.
	Derives from the base Config class and overrides some values.
	"""
	# Give the configuration a recognizable name
	NAME = "object"
	GPU_COUNT=1
	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 1
	# Number of classes (including background)
	NUM_CLASSES = 11  # Background + balloon
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 500
	USE_MINI_MASK = False
	WEIGHT_DECAY=0.001
	# Skip detections with < 80% confidence
	DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

	def load_balloon(self, dataset_dir, subset):
		"""Load a subset of the Balloon dataset.
		dataset_dir: Root directory of the dataset.
		subset: Subset to load: train or val
		"""
		# Add classes. We have only one class to add.
		classes = ['Hole(Virtual)','Hole(Physical)','Character Line Segment','Physical Degradation','Page Boundary','Character Component','Picture','Decorator','Library Marker']
		self.add_class("object", 1, "H-V")
		self.add_class("object", 2, "H")
		self.add_class("object", 3, "CLS")
		self.add_class("object", 4, "PD")
		self.add_class("object", 5, "PB")
		self.add_class("object", 6, "CC")
		self.add_class("object", 7, "P")
		self.add_class("object", 8, "D")
		self.add_class("object", 9, "LM")
		self.add_class("object", 10, "BL")

		# Train or validation dataset?
		assert subset in ["train", "val"]
		dataset_dir = os.path.join(dataset_dir, subset)

		# Load annotations
		# VGG Image Annotator (up to version 1.6) saves each image in the form:
		# { 'filename': '28503151_5b5b7ec140_b.jpg',
		#   'regions': {
		#       '0': {
		#           'region_attributes': {},
		#           'shape_attributes': {
		#               'all_points_x': [...],
		#               'all_points_y': [...],
		#               'name': 'polygon'}},
		#       ... more regions ...
		#   },
		#   'size': 100202
		# }
		# We mostly care about the x and y coordinates of each region
		# Note: In VIA 2.0, regions was changed from a dict to a list.
		annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
		annotations=	annotations["_via_img_metadata"]
		annotations = list(annotations.values())  # don't need the dict keys

		# The VIA tool saves images in the JSON even if they don't have any
		# annotations. Skip unannotated images.
		annotations = [a for a in annotations if a['regions']]

		# Add images
		for a in annotations:
			class_ids=[]
			# Get the x, y coordinaets of points of the polygons that make up
			# the outline of each object instance. These are stores in the
			# shape_attributes (see json format above)
			# The if condition is needed to support VIA versions 1.x and 2.x.
			if type(a['regions']) is dict:
				polygons = [r['shape_attributes'] for r in a['regions'].values()]
				objects = [s['region_attributes'] for s in a['regions'].values()]
			else:
				polygons = [r['shape_attributes'] for r in a['regions']]
				objects = [s['region_attributes'] for s in a['regions']] 

			#print(objects)
			classes = ['Hole(Virtual)','Hole(Physical)','Character Line Segment','Physical Degradation','Page Boundary','Character Component','Picture','Decorator','Library Marker']
			for obj in objects:
				if(obj['Spatial Annotation']=='Hole(Virtual)'):
					class_ids.append(1)
				if(obj['Spatial Annotation']=='Hole(Physical)'):
					class_ids.append(2)
				if(obj['Spatial Annotation']=='Character Line Segment'):
					class_ids.append(3)
				if(obj['Spatial Annotation']=='Physical Degradation'):
					class_ids.append(4)
				if(obj['Spatial Annotation']=='Page Boundary'):
					class_ids.append(5)
				if(obj['Spatial Annotation']=='Character Component'):
					class_ids.append(6)
				if(obj['Spatial Annotation']=='Picture'):
					class_ids.append(7)

				if(obj['Spatial Annotation']=='Decorator'):
					class_ids.append(8)

				if(obj['Spatial Annotation']=='Library Marker'):
					class_ids.append(9)
				if(obj['Spatial Annotation']=='Boundary Line'):	
					class_ids.append(10)


			# load_mask() needs the image size to convert polygons to masks.
			# Unfortunately, VIA doesn't include it in JSON, so we must read
			# the image. This is only managable since the dataset is tiny.

			ff=a['filename'].split('/')[-2:]
			#print(ff)
			flg=0
			if(ff[0]=='illustrations'):
				flg=0
				ff1=ff[0]+'/'+ff[1]
				image_path ='/home/anishg/prusty/'+ff1
 
			else:
				flg=1
				image_path=os.path.join(dataset_dir,a['filename'])
				image_pa = image_path.split("/")
				image_path = "/home/anishg/prusty/bhoomi_images/"
				flag=0
				for ppp in image_pa:
				 if(ppp=="images"):
					 flag=1
				 if(flag==1):
					 image_path=os.path.join(image_path,ppp)
				image_path=image_path.replace("%20"," " )
				image_path=image_path.replace("&","" )
			 #print(image_path)
			try:
				image = skimage.io.imread(image_path)
			except Exception:
				continue

			#image=cv2.resize(image,(1024,800))
			#image=cv2.resize(image,None,fx=1,fy=4,interpolation=cv2.INTER_CUBIC)
			height, width = image.shape[:2]

			self.add_image(
			 "object",
			 image_id=a['filename'],  # use file name as a unique image id
			 path=image_path,
			 width=width, height=height,
			 polygons=polygons,
			 num_ids=class_ids)

	
	def load_mask(self, image_id):
		"""Generate instance masks for an image.
	   Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "object":
			return super(self.__class__, self).load_mask(image_id)

		# Convert polygons to a bitmap mask of shape
		# [height, width, instance_count]
		info = self.image_info[image_id]
		num_ids = info['num_ids']
		mask = np.zeros( [info["height"], info["width"], len(info["polygons"])],
						dtype=np.uint8)
		for i, p in enumerate(info["polygons"]):
			# Get indexes of pixels inside the polygon and set them to 1
			try:
				rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
			except Exception:
				continue
			try:
				mask[rr, cc, i] = 1
				#mm=cv2.resize(mask[:,:,i],(0,0),fx=1,fy=4,interpolation=cv2.INTER_CUBIC)
				#print("mm: ",mm.shape)
				#mask[:,:,i]=mm[:info["height"],:]
			except Exception as e:
				print(e)

		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID only, we return an array of 1s
		num_ids = np.array(num_ids, dtype=np.int32)
		
		return mask.astype(np.bool), num_ids

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "object":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)


def train(model):
	"""Train the model."""
	# Training dataset.
	dataset_train = BalloonDataset()
	dataset_train.load_balloon(args.dataset, "train")
	dataset_train.prepare()

	# Validation dataset
	dataset_val = BalloonDataset()
	dataset_val.load_balloon(args.dataset, "val")
	dataset_val.prepare()

	# *** This training schedule is an example. Update to your needs ***
	# Since we're using a very small dataset, and starting from
	# COCO trained weights, we don't need to train too long. Also,
	# no need to train all layers, just the heads should do it.

	
	# augmentation = iaa.Sometimes(.667, iaa.Sequential([
	# 	# Strengthen or weaken the contrast in each image.
	# 	iaa.ContrastNormalization((0.75, 1.25)),
	# 	# Make some images brighter and some darker.
	# 	# In 20% of all cases, we sample the multiplier once per channel,
	# 	# which can end up changing the color of the images.
	# 	iaa.Multiply((0.8, 1.2)),
	# 	# Apply affine transformations to each image.
	# 	# Scale/zoom them, translate/move them, rotate them and shear them.
	# 	iaa.Affine(
	# 		scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
	# 		translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
	# 		rotate=(-10, 10),
	# 		shear=(-8, 8)
	# 	),
	# 	iaa.CoarseDropout(0.02, size_percent=0.5)	
	# ], random_order=True)) # apply augmenters in random order


	print("Training network heads")
	model.train(dataset_train, dataset_val,
						learning_rate=config.LEARNING_RATE,
						epochs=22,
						layers='heads')

	# Training - Stage 2
	# Finetune layers from ResNet stage 4 and up
	print("Fine tune Resnet stage 4 and up")
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=36,
				layers='4+')

	# Training - Stage 3
	# Fine tune all layers
	print("Fine tune all layers")
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE / 10,
				epochs=100,
				layers='all')


def color_splash(image, mask):
	"""Apply color splash effect.
	image: RGB image [height, width, 3]
	mask: instance segmentation mask [height, width, instance count]

	Returns result image.
	"""
	# Make a grayscale copy of the image. The grayscale copy still
	# has 3 RGB channels, though.
	gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
	# Copy color pixels from the original color image where mask is set
	if mask.shape[-1] > 0:
		# We're treating all instances as one, so collapse the mask into one layer
		mask = (np.sum(mask, -1, keepdims=True) >= 1)
		splash = np.where(mask, image, gray).astype(np.uint8)
	else:
		splash = gray.astype(np.uint8)
	return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
	assert image_path or video_path

	# Image or video?
	if image_path:
		# Run model detection and generate the color splash effect
		print("Running on {}".format(args.image))
		# Read image
		image = skimage.io.imread(args.image)
		# Detect objects
		r = model.detect([image], verbose=1)[0]
		# Color splash
		splash = color_splash(image, r['masks'])
		# Save output
		file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
		skimage.io.imsave(file_name, splash)
	elif video_path:
		import cv2
		# Video capture
		vcapture = cv2.VideoCapture(video_path)
		width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = vcapture.get(cv2.CAP_PROP_FPS)

		# Define codec and create video writer
		file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
		vwriter = cv2.VideoWriter(file_name,
								  cv2.VideoWriter_fourcc(*'MJPG'),
								  fps, (width, height))

		count = 0
		success = True
		while success:
			print("frame: ", count)
			# Read next image
			success, image = vcapture.read()
			if success:
				# OpenCV returns images as BGR, convert to RGB
				image = image[..., ::-1]
				# Detect objects
				r = model.detect([image], verbose=0)[0]
				# Color splash
				splash = color_splash(image, r['masks'])
				# RGB -> BGR to save image to video
				splash = splash[..., ::-1]
				# Add image to video writer
				vwriter.write(splash)
				count += 1
		vwriter.release()
	print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN to detect balloons.')
	parser.add_argument("command",
						metavar="<command>",
						help="'train' or 'splash'")
	parser.add_argument('--dataset', required=False,
						metavar="/path/to/balloon/dataset/",
						help='Directory of the Balloon dataset')
	parser.add_argument('--weights', required=True,
						metavar="/path/to/weights.h5",
						help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,
						default=DEFAULT_LOGS_DIR,
						metavar="/path/to/logs/",
						help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--image', required=False,
						metavar="path or URL to image",
						help='Image to apply the color splash effect on')
	parser.add_argument('--video', required=False,
						metavar="path or URL to video",
						help='Video to apply the color splash effect on')
	args = parser.parse_args()

	# Validate arguments
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "splash":
		assert args.image or args.video,\
			   "Provide --image or --video to apply color splash"

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)

	# Configurations
	if args.command == "train":
		config = BalloonConfig()
	else:
		class InferenceConfig(BalloonConfig):
			# Set batch size to 1 since we'll be running inference on
			# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
		config = InferenceConfig()
	config.display()

	# Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,
								  model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,
								  model_dir=args.logs)
	#model.save('trainedmrcnn.h5')
	# Select weights file to load
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		
		model.load_weights(weights_path, by_name=True, exclude=[
			"mrcnn_class_logits", "mrcnn_bbox_fc",
			"mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(weights_path, by_name=True)

	# Train or evaluate
	if args.command == "train":
		train(model)
	elif args.command == "splash":
		detect_and_color_splash(model, image_path=args.image,
								video_path=args.video)
	else:
		print("'{}' is not recognized. "
			  "Use 'train' or 'splash'".format(args.command))
