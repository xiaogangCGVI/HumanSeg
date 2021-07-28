import random
import math

import numpy as np 
import cv2

from PIL import Image

class ReColor:
    def __init__(self, alpha=0.1, beta=0.5):
        self._alpha = alpha
        self._beta = beta

    def __call__(self, im):
        # random amplify each channel
        t = np.random.uniform(-1, 1, 3)
        im *= (1 + t * self._alpha)
        mx = 255. * (1 + self._alpha)
        up = np.random.uniform(-1, 1)
        im = np.power(im / mx, 1. + up * self._beta)
        im = im * 255
        return im

class Resize:
    def __init__(self, im_size, mask_size=None):
        self.im_size = im_size
        if mask_size is None:
            mask_size = im_size
        self.mask_size = mask_size

    def __call__(self, im, mask=None):
        im = cv2.resize(im, self.im_size, interpolation=cv2.INTER_NEAREST)
        if mask is not None:
            mask = cv2.resize(mask, self.mask_size, interpolation=cv2.INTER_NEAREST)
            return im, mask
        return im

class ClipSize:
    def __init__(self, min_size=None, max_size=None):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, im, mask=None):
        if self.min_size is not None:
            min_edge = min(im.shape[0], im.shape[1])
            if min_edge < self.min_size:
                scale = 1.0*self.min_size/min_edge
                im = cv2.resize(im, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
                if mask is not None:
                    mask = cv2.resize(mask, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_NEAREST)
        if self.max_size is not None:
            max_edge = max(im.shape[0], im.shape[1])
            if max_edge > self.max_size:
                scale = 1.0*self.max_size/max_edge
                im = cv2.resize(im, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
                if mask is not None:
                    mask = cv2.resize(mask, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_NEAREST)
        return im, mask

class Padding:
    def __init__(self, factor=32):
        self.factor = 32

    def __call__(self, im, mask=None):
        im_h, im_w = im.shape[0], im.shape[1]
        pad_h = int(math.ceil(1.0*im_h/self.factor)*self.factor)
        pad_w = int(math.ceil(1.0*im_w/self.factor)*self.factor)
        pad_im = np.zeros((pad_h, pad_w, 3), im.dtype)
        pad_im[:im_h, :im_w, :] = im
        if mask is not None:
            pad_mask = np.zeros((pad_h, pad_w), mask.dtype)
            pad_mask[:im_h, :im_w] = mask
        else:
            pad_mask = None
        return pad_im, pad_mask

class RandomFlip:
    def __init__(self, mirror_map, axis=1):
        self.mirror_map = mirror_map
        self.axis = axis

    def __call__(self, im, mask):
        if random.random() > 0.5:
            if self.axis == 1:
                im = im[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            else:
                im = im[::-1, :, :].copy()
                mask = mask[::-1, :].copy()
            if self.mirror_map is not None:
                mask = mask[self.mirror_map, :]
        return im, mask

class RandomRotate:
    def __init__(self):
        pass

    def __call__(self, im, mask):
        rotate = random.randint(0, 3)
        img_h, img_w, _ = im.shape
        if rotate > 0:
            im = np.rot90(im, rotate)
            mask = np.rot90(im, mask)
        return im.copy(), mask.copy() 


class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, im, mask):
        x = random.randint(0, im.shape[1]-self.crop_size[0])
        y = random.randint(0, im.shape[0]-self.crop_size[1])
        crop_im = im[y:y+self.crop_size[1], x:x+self.crop_size[0], :]
        crop_mask = mask[y:y+self.crop_size[1], x:x+self.crop_size[0]]
        return crop_im, crop_mask


class Normalize:
    
    def __init__(self, scale=-1, mean=np.array([]), std=np.array([]), to_rgb=False):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb


    def __call__(self, im):
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.scale > 0:
            im *= self.scale
        if len(self.mean) != 0:
            im -= self.mean
        if len(self.std) != 0:
            im /= self.std
        return im

class Format:
    def __init__(self, mode):
        '''
        mode : pil2cv or cv2pil
        '''
        self.mode = mode

    def __call__(self, img):
        if self.mode == 'pil2cv':
            img = np.array(img, dtype=np.uint8)
            if len(img.shape) < 3:
                img = np.stack([img, img, img], axis=2)
            img = img[:,:,::-1] # rgb -> bgr
        if self.mode == 'cv2pil':
            img = img[:, :, ::-1]
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

