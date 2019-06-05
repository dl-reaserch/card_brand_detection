from mrcnn import utils
import os
from XmlDom import XmlParser
import numpy as np
from pycocotools import mask as maskUtils
import cv2

class CarBrandsDataset(utils.Dataset):
    def __init__(self,class_map=None):
        super().__init__()
        if "DATASET_MASK_TRAIN_PATH" in class_map:
            self.masks_path = class_map["DATASET_MASK_TRAIN_PATH"]
        elif "DATASET_MASK_VAL_PATH" in class_map:
            self.masks_path = class_map["DATASET_MASK_VAL_PATH"]
        self.parser = XmlParser()

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "car_brands":
            return super(CarBrandsDataset, self).load_mask(image_id)
        annotations = image_info['annotations']
        count = len(annotations)
        mask = np.zeros([image_info['height'],image_info['width'],count],
                        dtype=np.uint8)

        for i,annotation in enumerate(annotations):
            bbox = annotation['bbox']
            xmin,ymin,width,high = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            img = mask[:,:,i:i+1].copy()
            color = 1
            mask[:,:,i:i+1] = cv2.rectangle(img,(xmin,ymin),(xmin+width,ymin+high),color,-1)
        # Handle occlusions

        occlusion = np.logical_not(mask[:,:,-1].astype(np.uint8))
        for i in range(count-2,-1,-1):
            mask[:,:,i] = mask[:,:,i]*occlusion
            occlusion = np.logical_and(occlusion,np.logical_not(mask[:,:,i]))

        # Map class names to class IDs.
        class_ids = np.array([s['category_id'] for s in annotations])
        return mask.astype(np.bool),class_ids.astype(np.int32)




        # instance_masks = []
        # class_ids = []
        # annotations = self.image_info[image_id]["annotations"]
        # # Build mask of shape [height, width, instance_count] and list
        # # of class IDs that correspond to each channel of the mask.
        # for annotation in annotations:
        #     class_id = self.map_source_class_id(
        #         "car_brands.{}".format(annotation['category_id']))
        #     if class_id:
        #         m = self.annToMask(annotation, image_info["height"],
        #                            image_info["width"])
        #         # Some objects are so small that they're less than 1 pixel area
        #         # and end up rounded out. Skip those objects.
        #         if m.max() < 1:
        #             continue
        #         # Is it a crowd? If so, use a negative class ID.
        #         if annotation['iscrowd']:
        #             # Use negative class ID for crowds
        #             class_id *= -1
        #             # For crowd masks, annToMask() sometimes returns a mask
        #             # smaller than the given dimensions. If so, resize it.
        #             if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
        #                 m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
        #         instance_masks.append(m)
        #         class_ids.append(class_id)
        #
        # # Pack instance masks into an array
        # if class_ids:
        #     mask = np.stack(instance_masks, axis=2).astype(np.bool)
        #     class_ids = np.array(class_ids, dtype=np.int32)
        #     return mask, class_ids
        # else:
        #     # Call super class to return an empty mask
        #     return super(CarBrandsDataset, self).load_mask(image_id)


    def load_carTags(self,height,width):
        self.add_class("car_brands", 1,"benz")
        self.add_class("car_brands", 2, "toyota")

        class2id = {"benz":1,"toyota":2}
        if not os.path.exists(self.masks_path):
            raise RuntimeError('masks not found')

        files = os.listdir(self.masks_path)
        i = 0;
        for file_name in files:
            file_path = os.path.join(self.masks_path,file_name)
            #print("file_path:",file_path)
            self.parser.parse(file_path)
            boxes = self.parser.boxes
            anns = []
            for box in boxes:
                category_id = class2id[box[0]]
                bbox = [float(box[1]),float(box[2]),float(box[3])-float(box[1]),float(box[4])-float(box[2])]
                area = (float(box[3])-float(box[1]))*(float(box[4])-float(box[2]))
                print("category_id:",category_id)
                print("bbox:", bbox)
                print("area:", area)
                print("path:",self.parser.path)
                anns.append({"category_id": category_id,"bbox": bbox,"area": area})
            self.add_image("car_brands",
                           image_id=i,
                           path=self.parser.path,
                           width=width,
                           height=height,
                           annotations=anns
                           )
            i += 1


