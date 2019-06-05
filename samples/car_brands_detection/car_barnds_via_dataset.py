from mrcnn import utils
import os
import json
import skimage.io
import numpy as np
import  cv2
import matplotlib.pyplot as plt
from PIL import Image
from car_brands_config import CarBandsConfig
config = CarBandsConfig


class CarBrandsViaDataset(utils.Dataset):

    def load_car_brands(self,imgs_dir, marks_dir, subset):
        self.add_class("car_brands",1,"benz")
       # self.add_class("car_brands",2,"toyota")


        assert subset in ["train","val"]
        marks_dir = os.path.join(marks_dir, subset)
        imgs_dir = os.path.join(imgs_dir,subset)

        if not os.path.exists(marks_dir):
            raise RuntimeError("mark file not found")

        if not os.path.exists(imgs_dir):
            raise RuntimeError("img file not found")

        # Load annotations
        # VGG Image Annotator saves each image in the form:
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
        for filename in os.listdir(marks_dir):

            annotations = json.load(open(os.path.join(marks_dir, filename)))
            annotations = list(annotations.values())

            annotations = [a for a in annotations if a['regions']]
            # Add image

            for a in annotations:
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. There are stores in the
                # shape_attributes (see json format above)
                #polygons = [r['shape_attributes'] for r in a['regions'].values()]

                shape_attributes = [r['shape_attributes'] for r in a['regions'].values()]
                image_path = os.path.join(imgs_dir, a['filename'])
                #print(image_path)
                classes = [r['region_attributes']['name'] for r in a['regions'].values()]
                image = skimage.io.imread(image_path)
                height,width = image.shape[:2]
                self.add_image(
                    "car_brands",
                    image_id=a['filename'],
                    path=image_path,
                    width=width,
                    height=height,
                    classes=classes,
                    shape_attributes=shape_attributes
                )


    def load_mask(self, image_id):
        """Generate instance masks for an image.
               Returns:
                masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
                """
        # If not a balloon dataset image, delegate to parent class.

        image_info = self.image_info[image_id]
        if image_info['source'] != "car_brands":
            return super(self.__class__,self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        # info = self.image_info[image_id]
        # mask = np.zeros([info['height'],info['width'],len(info["polygons"])],dtype=np.uint8)
        # for i, p in enumerate(info['polygons']):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'],p['all_points_x'])
        #     mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shape_attributes"])],
                        dtype=np.uint8)

        l = len(info["shape_attributes"])
        for i, p in enumerate(info["shape_attributes"]):
            shape = p['name']
            if "ellipse" == shape:
                #椭圆
                r = p['cx']
                c = p['cy']
                r_radius = p['rx']
                c_radius = p['ry']
                rr, cc = skimage.draw.ellipse(r, c, r_radius, c_radius,shape=mask.shape)
                #print("rr:", rr.shape, "--cc:", cc.shape, "--mask.shape:", mask.shape)
                mask[rr, cc, i] = 1
            elif "circle" == shape:
                #圆
                r = p['cx']
                c = p['cy']
                radius = p['r']
                rr, cc = skimage.draw.circle(r,c,radius,shape=mask.shape)
                #print("rr:", rr.shape, "--cc:", cc.shape, "--mask.shape:", mask.shape)
                mask[rr, cc, i] = 1
            elif "polygon" == shape:
                #多边形
                rr, cc = skimage.draw.polygon(np.array(p['all_points_y']), np.array(p['all_points_x']),shape=mask.shape)
                #print("rr:", rr.shape, "--cc:", cc.shape, "--mask.shape:", mask.shape)
                mask[rr, cc, i] = 1
            elif "rect"== shape:
                pass
                #长方形
                # start =(p['x'],p['y'])
                # extent = (p['width'],p['height'])
                # rr, cc = skimage.draw.rectangle(start,extent)
                # # cv2.rectangle(mask[cc,cc,0], (x - s, y - s), (x + s, y + s), color, -1)
                # # cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
                # mask[rr, cc, i] = 1
            # Get indexes of pixels inside the polygon and set them to 1

            #classes[]

        # Handle occlusions
        # height = 512
        # width = 512
        # #( 1,512,512,3)
        # x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.
        # print("x_shape:", x.shape)
        # x = x.reshape((height, width, 3)) #(512,512,3)
        # x = x[:, :, ::-1]
        # x[:, :, 0] += 103.939
        # x[:, :, 1] += 116.779
        # x[:, :, 2] += 123.68
        # x = np.clip(x, 0, 255).astype('uint8')
        # newImg = Image.fromarray(x)


        # occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        # for i in range(l - 2, -1, -1):
        #     mask[:, :, i] = mask[:, :, i] * occlusion
        #     occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # plt.imshow(mask.reshape(25,25), cmap='Greys_r')
        # plt.show()
        # x = np.clip(mask, 0, 255).astype('uint8')
        # newImg = Image.fromarray(x)
        # Image._show(newImg)

        # molded_image, window, scale, padding, crop = utils.resize_image(
        #     mask,
        #     min_dim=config.IMAGE_MIN_DIM,
        #     min_scale=config.IMAGE_MIN_SCALE,
        #     max_dim=config.IMAGE_MAX_DIM,
        #     mode=config.IMAGE_RESIZE_MODE)
        # #molded_image = mask(molded_image,config)
        # x = np.clip(molded_image, 0, 255).astype('uint8')
        # newImg = Image.fromarray(x)
        # Image._show(newImg)


        class_ids = np.array([self.class_names.index(s) for s in info['classes']])

        return mask.astype(np.bool),class_ids.astype(np.int32)

