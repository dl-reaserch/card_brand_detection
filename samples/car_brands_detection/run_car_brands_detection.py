import os
import sys
import numpy as np
import imgaug

ROOT_DIR = os.path.abspath("../../")
APP_DIR = os.path.join(ROOT_DIR,"samples/car_brands_detection")
APP_NAME = "car_brands"
DATASET_ROOT_DIR = "D:\Repo\AI\Datasets"

DATASET_APP_DIR = os.path.join(DATASET_ROOT_DIR, APP_NAME)
# DATASET_MASK_TRAIN_PATH = os.path.join(DATASET_ROOT_DIR, "train_car_masks")
# DATASET_MASK_VAL_PATH = os.path.join(DATASET_ROOT_DIR, "val_car_masks")
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from car_brands_config import CarBandsConfig
from car_brands_dataset import CarBrandsDataset
from car_barnds_via_dataset import CarBrandsViaDataset

MODEL_DIR = os.path.join(APP_DIR, "logs")

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = CarBandsConfig()
# class_map = {"DATASET_MASK_TRAIN_PATH":DATASET_MASK_TRAIN_PATH}
# train_data_set = CarBrandsDataset(class_map)
# train_data_set.load_carTags(config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
# train_data_set.prepare()
#
# class_map ={"DATASET_MASK_VAL_PATH":DATASET_MASK_VAL_PATH}
# val_data_set = CarBrandsDataset(class_map)
# val_data_set.load_carTags(config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
# val_data_set.prepare()


#image_ids = np.random.choice(data_set.image_ids,4)

# for image_id in image_ids:
#     image = data_set.load_image(image_id)
#     mask,class_ids = data_set.load_mask(image_id)

DATASET_APP_IMG_DIR = os.path.join(DATASET_APP_DIR,"imgs")
DATASET_APP_MARKS_DIR = os.path.join(DATASET_APP_DIR,"marks")

data_train = CarBrandsViaDataset()
data_train.load_car_brands(DATASET_APP_IMG_DIR,DATASET_APP_MARKS_DIR, "train")
data_train.prepare()

data_val = CarBrandsViaDataset()
data_val.load_car_brands(DATASET_APP_IMG_DIR,DATASET_APP_MARKS_DIR, "val")
data_val.prepare()
model = modellib.MaskRCNN(mode="training",config=config,model_dir=MODEL_DIR)

init_with ="last"
if init_with =="imagenet":
    model.load_weights(model.get_imagenet_weights(),by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH,by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"]
                       )
elif init_with == "last":
    print("latest:",model.find_last()[1])
    model.load_weights(model.find_last()[1],by_name=True)

augmentation = imgaug.augmenters.Sometimes(0.5, [
                imgaug.augmenters.Fliplr(0.5),
                imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
            ])
model.train(data_train,data_val,learning_rate=config.LEARNING_RATE,epochs=30,layers='heads',augmentation=augmentation)

