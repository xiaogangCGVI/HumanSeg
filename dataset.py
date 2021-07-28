# extern
import PIL.Image as Image
import cv2
import numpy as np
import numpy.random as random
import supervisely_lib as sly

# torch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter

from utils import generate_mask_path
from preprocess import Resize, RandomFlip, Normalize, Format


class HumanDataset(Dataset):
    
    def __init__(self, img_list, is_train=False):
        self.is_train = is_train
        # load list
        f = open(img_list, 'r') 
        self.seg_imgs = [line.strip() for line in f if len(line)>2]        
        self.seg_labels = [generate_mask_path(p, True) for p in self.seg_imgs]

        mean = np.array([[[123.675, 116.28, 103.53]]], dtype=np.float32)
        std = np.array([[[58.395, 57.12, 57.375]]], dtype=np.float32)

        if is_train:
            self.trans_color = [ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.3),
                                Format('pil2cv')]
            self.trans_normal = [Normalize(-1, mean, std, True)]
            self.trans_all = [
                    Resize(im_size=(512, 512)),
                    RandomFlip(mirror_map=None)]
        else:
            self.trans_color = []
            self.trans_normal = [Normalize(-1, mean, std, True)]
            self.trans_all = [
                    Resize(im_size=(512,512))]
            

    def __getitem__(self, index):
        # load data
        img_path = self.seg_imgs[index]
        mask_path = self.seg_labels[index]
        if self.is_train:
            img = Image.open(img_path)
        else:
            img = cv2.imread(img_path)
        # TODO: add edge learning
        mask = cv2.imread(mask_path)
        mask = mask / 255.0

        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        
        # preprocess color image
        for trans in self.trans_color:
            img = trans(img)
        img = img.astype(np.float32)
        # normalize
        for trans in self.trans_normal:
            img = trans(img)
        for trans in self.trans_all:
            img, mask = trans(img, mask)
        '''
        mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                interpolation=cv2.INTER_NEAREST)
        '''
        img = torch.Tensor(img).float().permute(2, 0, 1)
        mask = torch.Tensor(mask).float().unsqueeze(0) # long
        return img, mask

    def __len__(self):
        return len(self.seg_labels)


class MattingDataset(Dataset):
    
    def __init__(self, img_list, is_train=False):
        self.is_train = is_train
        # load list
        f = open(img_list, 'r') 
        self.seg_imgs = [line.strip() for line in f if len(line)>2]        
        self.seg_labels = [generate_mask_path(p, True, 'matting') for p in self.seg_imgs]

        mean = np.array([[[123.675, 116.28, 103.53]]], dtype=np.float32)
        std = np.array([[[58.395, 57.12, 57.375]]], dtype=np.float32)

        if is_train:
            self.trans_color = [ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.3),
                                Format('pil2cv')]
            self.trans_normal = [Normalize(-1, mean, std, True)]
            self.trans_all = [
                    Resize(im_size=(512, 512)),
                    RandomFlip(mirror_map=None)]
        else:
            self.trans_color = []
            self.trans_normal = [Normalize(-1, mean, std, True)]
            self.trans_all = [
                    Resize(im_size=(512,512))]
            

    def __getitem__(self, index):
        # load data
        img_path = self.seg_imgs[index]
        mask_path = self.seg_labels[index]
        if self.is_train:
            img = Image.open(img_path)
        else:
            img = cv2.imread(img_path)
        # TODO: add edge learning
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = mask[:, :, 3] / 255.0
        
        # preprocess color image
        for trans in self.trans_color:
            img = trans(img)
        img = img.astype(np.float32)
        # normalize
        for trans in self.trans_normal:
            img = trans(img)
        for trans in self.trans_all:
            img, mask = trans(img, mask)
        '''
        mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                interpolation=cv2.INTER_NEAREST)
        '''
        img = torch.Tensor(img).float().permute(2, 0, 1)
        mask = torch.Tensor(mask).float().unsqueeze(0) # long
        return img, mask

    def __len__(self):
        return len(self.seg_labels)


if __name__ == '__main__':
    #dataset = HumanDataset('/Users/tiangang.zhang/work/data/SuperviselyPersonDataset/train.txt', is_train=True)
    dataset = MattingDataset('/Users/tiangang.zhang/mnt/data/SDE/dataset/human_half/train.txt', is_train=True)
    n = len(dataset)
    while True:
        index = random.randint(0, n)
        img, label = dataset[index]
        label = np.stack([label, label, label], axis=2).astype(np.uint8)
        vis_img = np.concatenate((img.astype(np.uint8), label), axis=1)
        cv2.imshow('demo', vis_img)
        k = cv2.waitKey()
        if k == ord('q'):
            break

