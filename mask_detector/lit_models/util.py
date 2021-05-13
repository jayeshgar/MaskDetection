from typing import Union
import torch
from bs4 import BeautifulSoup
import os
import torch.nn as nn
from mask_detector.models.util import *
from pytorch_lightning.callbacks import Callback
from mask_detector.data.util import prep_image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import cv2
from google.colab.patches import cv2_imshow
#from google.colab.patches import cv2_imshow # for image display

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
    targets = [torch.cat([target["boxes"],target["labels"].view((-1,1))],1) for target in y]
    # Final targets will carry the targets specifically for each logit result
    #So shape should be as that of logits
    finals = torch.zeros(logits.shape)
    if CUDA:
        logits = logits.cuda()
        finals = finals.cuda()
        targets = [target.cuda() for target in targets]
    
    #Transform the center and width/height coordinates to log left and bottom right coordinates
    box_corner = logits.new(logits.shape)
    box_corner[:,:,0] = (logits[:,:,0] - logits[:,:,2]/2)
    box_corner[:,:,1] = (logits[:,:,1] - logits[:,:,3]/2)
    box_corner[:,:,2] = (logits[:,:,0] + logits[:,:,2]/2) 
    box_corner[:,:,3] = (logits[:,:,1] + logits[:,:,3]/2)
    logits[:,:,:4] = box_corner[:,:,:4]

    for ind in range(batch_size):        
        boxes = targets[ind]
        for index,box in enumerate(boxes):
            predictions = torch.clone(logits[ind])          #image Tensor
            #Get the IOUs of all boxes that come after the one we are looking at 
            #in the loop
            ious = bbox_iou(box.unsqueeze(0), predictions)
            #Zero out all the detections that have IoU != 0
            iou_mask = (ious != 0).float().unsqueeze(1)
            predictions *= iou_mask       
            
            #Remove the non-zero entries
            non_zero_ind = torch.nonzero(predictions[:,4]).squeeze()

            #Setting the finals to be passed for metrics 
            finals[ind][non_zero_ind, 4] = 1
            #Taking from image
            zero_ind = torch.tensor([not_idx for not_idx in range(predictions.size(0)) if not_idx not in non_zero_ind])
            no_predictions = predictions[zero_ind]
            predictions = predictions[non_zero_ind]
            box_0 = torch.repeat_interleave(box[0],repeats=predictions.size(0))
            box_1 = torch.repeat_interleave(box[1],repeats=predictions.size(0))
            box_2 = torch.repeat_interleave(box[2],repeats=predictions.size(0))
            box_3 = torch.repeat_interleave(box[3],repeats=predictions.size(0))
            box_4 = torch.repeat_interleave(box[4],repeats=predictions.size(0))            
            loss_x = lambda_coord * mse_loss(predictions[:,0],box_0)
            print("loss_x = ", loss_x)
            loss_y = lambda_coord * mse_loss(predictions[:,1],box_1)
            print("loss_y = ", loss_y)
            loss_w = lambda_coord * mse_loss(predictions[:,2] - predictions[:,0],box_2-box_0)
            print("loss_w = ", loss_w)
            loss_h = lambda_coord * mse_loss(predictions[:,3] - predictions[:,1],box_3-box_1)
            print("loss_h = ", loss_h)
            expected_conf = torch.ones(predictions.size(0)) 
            expected_noconf = torch.zeros(logits[ind].size(0) - predictions.size(0))
            if CUDA:
                expected_conf = expected_conf.cuda()
                expected_noconf = expected_noconf.cuda()
            loss_conf =  mse_loss(predictions[:,4], expected_conf)  
            print("loss_conf = ", loss_conf)          
            loss_noconf =  lambda_noobj*mse_loss(no_predictions[:,4], expected_noconf)
            print("loss_noconf = ", loss_noconf)
            loss_cls = (1 / batch_size) * ce_loss(predictions[:,5:], box_4.long())
            print("loss_cls = ", loss_cls)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_noconf + loss_cls            
        total_loss = total_loss + loss
        
    return total_loss,logits,finals


class SanityCheckCallback(Callback):
    
    images_dir = "mask_detector/data/sample_images/"
    imgs = list(sorted(os.listdir(images_dir)))
    labels_dir = "mask_detector/data/sample_annotations/"
    labels = list(sorted(os.listdir(labels_dir)))

    def print_samples(self):
        for image,annotation in zip(self.imgs,self.labels):
            img = Image.open(self.images_dir+image).convert("RGB")
            target = generate_target(0,self.labels_dir+annotation)
            self.plot_image(img, target)

    def sanity_test(self,trainer,pl_module):
        num_classes = 3
        confidence = 0.5
        for image,annotation in zip(self.imgs,self.labels):
            img = Image.open(self.images_dir+image).convert("RGB")
            img = np.array(img) #Convert into numpy  array
            target = generate_target(0,self.labels_dir+annotation)
            #prepare the image for model input
            target_img_size = 416  #Target image size as per yolo
            img = prep_image(img, target_img_size).squeeze()
            img = img.permute(1, 2, 0)
            fig,ax = plt.subplots(1)
            #ax.imshow(img.permute(1, 2, 0))
            #plt.show()
            cv2_imshow(img.cpu().detach().numpy())
            #Fetch the model output
            #if torch.cuda.is_available():
            #    image = image.cuda()
            #logit = pl_module.forward(image.unsqueeze(0))
            #logit = write_results(logit, confidence, num_classes, nms_conf = 0.4)
            #print("logit shape = ",logit.shape)
            #boxes = logit[:,:4]
            #print("boxes shape = ", boxes.shape)
            #labels = torch.argmax(logit[:,5:],dim=1)
            #print("labels shape = ", labels.shape)
            #Format the boxes of the target
            #img_orig_dim = [img.shape[0],img.shape[1]]
            #img_w, img_h = img_orig_dim[1], img_orig_dim[0]
            #new_w = int(img_w * min(target_img_size/img_w, target_img_size/img_h))
            #new_h = int(img_h * min(target_img_size/img_w, target_img_size/img_h))
            #Remove the unnecessary boxes 
            #boxes[:,[0,2]] = (boxes[:,[0,2]] - (target_img_size-new_w)//2) /min(target_img_size/img_w, target_img_size/img_h)
            #boxes[:,[1,3]] = (boxes[:,[1,3]] - (target_img_size-new_h)//2) /min(target_img_size/img_w, target_img_size/img_h)
            #target["boxes"] = boxes
            #target["labels"] = labels
            #self.plot_image(img, target)

    def plot_image(self,imgs, annotations):
        fig,ax = plt.subplots(1)
        ax.imshow(img)
        #cv2_imshow(img)
        labels = annotation["labels"]
        for index,box in enumerate(annotation["boxes"]):        
            xmin, ymin, xmax, ymax = box
            # Create a Rectangle patch
            if labels[index] == 0:
                color = 'r'
            if labels[index] == 1:
                color = 'g'
            if labels[index] == 2:
                color = 'y'
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor=color,facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()
    
    def on_train_start(self, trainer, pl_module):
        print('Run the sanity check when the training starts')
        self.sanity_test(trainer, pl_module)
        

    def on_train_end(self, trainer, pl_module):
        print('Run the sanity check when training ends')
        self.sanity_test(trainer, pl_module)