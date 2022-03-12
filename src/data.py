import torch
import torch.nn as nn
import os
import numpy as np
import pandas 
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from src.utils import cls_init
from pycocotools.coco import COCO
import logging 
import sys

logger = logging.getLogger(__name__)

__all__ = ["COCODataset"]


class COCODataset(Dataset):
    def __init__(self, img_dir, metafile, mode, debug, classes):
        super(COCODataset, self).__init__()
        self.metafile = metafile
        self.img_dir = img_dir
        self.debug = debug
        self.mode = mode
        self.targets = classes    
        self.coco = COCO(metafile)
        self.transforms = self._transforms_img(mode)  
        self.load_classes()    
        self.image_ids = self._get_img_ids(self.metafile, self.coco, self.debug, self.targets)
        
        # logger.info(f"LOADED {self.mode} dataset")

    def _transforms_img(self, mode):
        normalize = transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )

        # define transforms
        transform = transforms.Compose([
                transforms.Resize((264, 264)),
                transforms.ToTensor(),
                normalize,
            ])
        
        return transform
        # transform = transforms.Compose([
        #     transforms.Lambda(lambda img: img.copy()),
        #     transforms.ToTensor(), 
        #     utils.normalize_transform()
        # ])
    # transform = A.Compose([
#     A.RandomCrop(width=450, height=450),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ], bbox_params=A.BboxParams(format='coco'))

    def _get_img_ids(self, metafile, coco, debug, targets):        
        ids = list(sorted(coco.imgs.keys()))
        category_ids = coco.getCatIds(targets)
        image_ids = coco.getImgIds(catIds=category_ids)
        if debug:
            image_ids = image_ids[:2]
        return image_ids

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.image_ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.img_dir, path))
        num_objs = len(coco_annotation)

        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            if coco_annotation[i]["category_id"] != self.class_2_label(self.targets):
                continue
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = torch.ones((num_objs,), dtype=torch.int64)        
        img_id = torch.tensor([img_id])
        
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
        
        # tr_img = img.copy()
        # anno_img = img[::-1]
        # for i in range(len(boxes)):
        #      # (x1, y1, w, h)
        #     x, y, w, h = boxes[i]          
        
        #     anno_image = cv2.rectangle(anno_img, (int(x), int(y)), (int(w), int(h)), (0, 255, 255), 2) 

        # cv2.imshow("demo", anno_img)
        # cv2.waitKey(0)

        return img,  my_annotation
     

    def __len__(self):
        return len(self.image_ids)

    def load_classes(self):
            """ Loads the class to label mapping (and inverse) for COCO.
            """
            # load class names (name -> label)
            categories = self.coco.loadCats(self.coco.getCatIds())
            categories.sort(key=lambda x: x['id'])

            self.classes             = {}
            self.coco_labels         = {}
            self.coco_labels_inverse = {}
            for c in categories:
                self.coco_labels[len(self.classes)] = c['id']
                self.coco_labels_inverse[c['id']] = len(self.classes)
                self.classes[c['name']] = len(self.classes) + 1

            # also load the reverse (label -> name)
            self.labels = {}
            for key, value in self.classes.items():
                self.labels[value] = key

    def class_2_label(self, coco_classes):
        return self.classes[coco_classes]
    
def collate_fn(batch):
   return tuple(zip(*batch))

def get_loaders(train_cls, val_cls, batch_size):
    train_ds = cls_init(train_cls)
    val_ds = cls_init(val_cls)

    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    return train_dl, val_dl


