from typing import Union
import torch
from bs4 import BeautifulSoup
import os
import torch.nn as nn
from mask_detector.models.util import *

def generate_box(obj):
    
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

def generate_target(image_id, file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        
        return target

def yolo_loss(logits,y, CUDA = True):
    lambda_coord = 5
    lambda_noobj = 0.5
    mse_loss = nn.MSELoss(reduction='mean')
    ce_loss = nn.CrossEntropyLoss() 
    total_loss = 0
    batch_size = logits.size(0)
    num_classes = 3
    confidence = 0.5
    results = [target["boxes"] for target in y]  
    expected_conf = torch.FloatTensor(1)  
    if CUDA:
        logits = logits.cuda()
        results = [target["boxes"].cuda() for target in y]
        expected_conf = expected_conf.cuda()
    logits = train_iou(logits, confidence, results)
    targets = torch.cat([torch.cat([target["boxes"],target["labels"].view((-1,1))],1) for target in y])
    loss_x = lambda_coord * mse_loss(logits[:,0],targets[:,0])
    loss_y = lambda_coord * mse_loss(logits[:,1],targets[:,1])
    loss_w = lambda_coord * mse_loss(logits[:,2] - logits[:,0],targets[:,2]-targets[:,0])
    loss_h = lambda_coord * mse_loss(logits[:,3] - logits[:,1],targets[:,3]-targets[:,1])        
    loss_conf =  mse_loss(logits[:,4], expected_conf)
    loss_cls = (1 / batch_size) * ce_loss(logits[:,5:], targets[:,4].long())
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
    total_loss = total_loss + loss
    return total_loss,logits,targets