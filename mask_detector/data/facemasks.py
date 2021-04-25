"""FaceMasks DataModule"""
import argparse
import os
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms
from PIL import Image
from mask_detector.data.base_data_module import BaseDataModule, load_and_print_info
from mask_detector.data.util import collate_fn,generate_target,prep_image
import numpy as np

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname()

class MaskDataset(object):
    def __init__(self, transforms, args):
        #super().__init__(args)
        self.args = args
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        data_path = self.args["data_path"]+"/"
        self.imgs = list(sorted(os.listdir(data_path+"images/")))
#         self.labels = list(sorted(os.listdir("/kaggle/input/face-mask-detection/annotations/")))

    def __getitem__(self, idx):
        # load images ad masks
        file_image = 'maksssksksss'+ str(idx) + '.png'
        file_label = 'maksssksksss'+ str(idx) + '.xml'
        data_path = self.args["data_path"]+"/"
        img_path = os.path.join(data_path+"images/", file_image)
        label_path = os.path.join(data_path+"annotations/", file_label)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img) #Convert into numpy  array
        #Keep image dimension in a different location
        img_orig_dim = [img.shape[0],img.shape[1]]
        target_img_size = 416  #Target image size as per yolo
        img = prep_image(img, target_img_size).squeeze()
        #print("img shape = ",img.shape)
        #img = Image.fromarray(img)
        #Generate Label
        target = generate_target(idx, label_path)
        target["img_size"] = img_orig_dim
        #print("target shape = ",target)
        #if self.transforms is not None:
        #    img = self.transforms(img) 
        #Return only the image id. Fetching  the exact label
        # And calculating the loss with be taken care later 
        # At loss calculation level               
        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def add_to_argparse(parser):
        print("Adding model arguments")
        parser.add_argument("--data_path", dest = "data_path", help = "Data folder where images and annotations are stored", default = '/content/drive/MyDrive/data/MaskDetection/data')
        return parser


class FACEMASKS(BaseDataModule):
    """
    facemasks DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = vars(args) if args is not None else {}
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.transform = transforms.Compose([transforms.ToTensor()])
        #self.dims = (1, 28, 28)  # dims are returned when calling `.size()` on this object.
        #self.output_dims = (1,)
        self.mapping = list(range(10))

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Split into train, val, test, and set dims."""
        #facemasks_full = TorchMNIST(self.data_dir, train=True, transform=self.transform)
        #self.data_train, self.data_val = random_split(facemasks_full, [55000, 5000])
        #self.data_test = TorchMNIST(self.data_dir, train=False, transform=self.transform)
        dataset = MaskDataset(self.transform,self.args)
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.20)
        val_idx, test_idx = train_test_split(val_idx, test_size=0.5)
        self.data_train = Subset(dataset, train_idx)
        self.data_val = Subset(dataset, val_idx)
        self.data_test = Subset(dataset, test_idx)
        print("size of train set = ",len(self.data_train),", val set = ",len(self.data_val),", test set = ",len(self.data_test))
        #self.data_train = torch.utils.data.DataLoader(self.data_train, batch_size=4, collate_fn=collate_fn)
        #self.data_val = torch.utils.data.DataLoader(self.data_val, batch_size=4, collate_fn=collate_fn)
        #self.data_test = torch.utils.data.DataLoader(self.data_test, batch_size=4, collate_fn=collate_fn)
        


if __name__ == "__main__":
    load_and_print_info(FACEMASKS)
