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
    for target,logit in zip(y,logits):
        #file_label = 'maksssksksss'+ str(label[0].item()) + '.xml'
        #label_path = os.path.join("C:/Practice/POC/MaskDetection/mask_detector/data/annotations/", file_label)
        #target = y
        #target returns xmin,ymin,xmax,ymax
        #target with_mask - 1, mask_weared_incorrect - 2,without_mask - 0
        #logit returns xmin,ymin,xmax,ymax 
        #target =  {'boxes': tensor([[ 96., 226., 199., 350.]]), 'labels': tensor([1]), 'image_id': tensor([403])}
        #logit =  tensor([  0.0000, 316.1546, -11.3081, 331.9244,  19.5196,   0.5157,   0.5100,
        #  0.0000])
        #print("target = ",target)
        #Apply write_results to reduce the boxes even more
        #print("logit initially = ",logit.shape)
        logit = write_results(logit.unsqueeze(1), confidence, num_classes, nms_conf = 0.4)
        #print("logit = ",logit.shape)
        boxes = target["boxes"]
        boxes = torch.squeeze(boxes,dim=1)
        #Rescaling the image as the logits are working in a different dimension
        img_w, img_h = target["img_size"][1], target["img_size"][0]
        w = 416
        h = 416
        new_w = int(img_w * min(w/img_w, h/img_h))
        new_h = int(img_h * min(w/img_w, h/img_h))
        boxes[:,0] = (w-new_w)//2 + boxes[:,0]*min(w/img_w, h/img_h)
        boxes[:,2] = (w-new_w)//2 + boxes[:,2]*min(w/img_w, h/img_h)
        boxes[:,1] = (h-new_h)//2 + boxes[:,1]*min(w/img_w, h/img_h)
        boxes[:,3] = (h-new_h)//2 + boxes[:,3]*min(w/img_w, h/img_h)
        for index,box in enumerate(boxes):
            target_cls = target["labels"][index]
            loss_x = lambda_coord * mse_loss(logit[index][1],box[0])
            loss_y = lambda_coord * mse_loss(logit[index][2],box[1])
            loss_w = lambda_coord * mse_loss(logit[index][3] - logit[index][1],box[2]-box[0])
            loss_h = lambda_coord * mse_loss(logit[index][4] - logit[index][2],box[3]-box[1])
            expected_conf = torch.FloatTensor(1)
            input_logit = torch.zeros(1,3)
            if CUDA:
                logit = logit.cuda()
                expected_conf = expected_conf.cuda()
                input_logit = input_logit.cuda()
            loss_conf =  mse_loss(logit[index][5], expected_conf)
            #loss_conf = loss_conf + lambda_noobj * mse_loss(logit[index][5], expected_conf)            
            print("input_logit = ",input_logit,",logit[index][7].long() = ",logit[index][7].long())
            input_logit[logit[index][7].long()] = 1
            print("batch_size = ",batch_size,",input_logit = ",input_logit,"target_cls = ",target_cls)
            loss_cls = (1 / batch_size) * ce_loss(input_logit, target_cls.unsqueeze(0))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            total_loss = total_loss + loss
        #One more to account for the number of boxes generated and what was actually targetted
        loss_noobj = lambda_noobj * mse_loss(logit.shape[0], boxes.shape[0])
        total_loss += loss_noobj
    return total_loss

def getTensors(logits,targets, CUDA = True):
    num_classes = 3
    confidence = 0.5
    target_out = []
    logit_out = []
    for logit,target in zip(logits,targets):
        
        target_temp = target["labels"]        
        target_shape = target["labels"].shape[0]
        #print("target shape = ",target_shape)
        logit_temp = write_results(logit.unsqueeze(1), confidence, num_classes, nms_conf = 0.4)[:,7][0:target_shape]
        #Compare the target with the logit values
        logit_temp = torch.bitwise_xor(target_temp,logit_temp.long())
        logit_temp = torch.count_nonzero(logit_temp)
        logit_temp = torch.tensor(target_shape) - logit_temp
        target_out.append(torch.tensor(target_shape).unsqueeze(0))
        logit_out.append(logit_temp.unsqueeze(0))
    logit_final = torch.cat(logit_out)
    target_final = torch.cat(target_out)
    if CUDA:
        logit_final = logit_final.cuda()
        target_final = target_final.cuda()
    return logit_final,target_final