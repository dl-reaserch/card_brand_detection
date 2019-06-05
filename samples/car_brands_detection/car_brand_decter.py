from car_brands_config import CarBandsConfig
import mrcnn.model as modellib
import os
import skimage.io
from mrcnn import visualize
ROOT_DIR = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMG_DIR = os.path.join(ROOT_DIR,"imgs")
print(ROOT_DIR)
class InferenceConfig(CarBandsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
   # IMAGE_SHAPE = (64,64)

inference_config = InferenceConfig()

#MODEL_DIR = os.path.join(ROOT_DIR, "logs")

model = modellib.MaskRCNN(mode="inference",config=inference_config,model_dir=MODEL_DIR)
model_path = model.find_last()[1]

assert model_path !="","Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path,by_name=True)

class_names=['BG','benz','toyota']
image = skimage.io.imread(os.path.join(IMG_DIR,"tytfxp02.jpg"))
result = model.detect([image],verbose=1)
r = result[0]
visualize.display_instances(image,r['rois'],r['masks'],r['class_ids'],class_names,r['scores'])

#car_tags.png
