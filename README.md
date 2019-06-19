# Region-segmentation

This system takes in an input image and outputs a image with region labels overlaid on top of the image. It also generates a json which can then be loaded as a project in the annotator tool for further refinement. We also provide instructions for training the model.

### Install prerequisites 

```bash
python3 -m pip install -r requirements.txt
```

### To run Inference on your own image

1. Download the pretrained model from this [link](https://drive.google.com/file/d/1TFUEjo4D7een7C7fGJV-xrU1cKi_hFeO/view?usp=sharing) 
2. Place the `pretrained_model_indiscapes.h5` file in the root folder (`Instance-segmentation`)
3. Start the  GUI application (`Instance-segmentation/main/doc/app.py`)
```bash
python3 app.py
```
If you get an error "No module Named skimage". It can be fixed by installing scikit-image. Enter- pip install scikit-image in your Terminal or Command Prompt.

4. Upload the image and click on submit. To generate the json, click on `Create json`. This will generate a json which can be loaded in the annotator.

![app](/images/app.png)

5. This also outputs the final result (Labelled regions overlaid on top of the original image) at `Instance-segmentation/main/doc/static/images/2.jpg`

![Results](/images/result.png)

### To train the model

1. Download the Indiscapes dataset from this [link](http://ihdia.iiit.ac.in/indiscapes/) and `mask_rcnn_coco.h5
`from this [link](https://github.com/matterport/Mask_RCNN/releases)
2. Place the folders `bhoomi_images` and `PIH_images` and the file `mask_rcnn_coco.h5` inside the root folder (`Instance-segmentation`)
3. To start training :
   - Train a new model starting from pre-trained COCO weights
```bash
	python3 train.py train --dataset=/path/to/doc/dataset --weights=coco
```

   - Resume training a model that you had trained earlier
```bash
	python3 train.py train --dataset=/path/to/doc/dataset --weights=last
```
