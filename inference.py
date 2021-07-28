import cv2
import torch
import numpy as np

from preprocess import Normalize, Resize

class InferenceWrapper(object):

    def __init__(self, net, model_file, with_cuda, thresh=0.5):
        device = torch.device('cuda:0') if with_cuda else torch.device('cpu')
        self.thresh = thresh
        self.with_cuda = with_cuda

        # load weight
        # 1. .to() transfer weight tensor to gpu (cpu no need .to())
        # 2. map_location transfer model to gpu
        if with_cuda:
            net.load_state_dict(torch.load(model_file))
            net.to(device)
        else:
            net.load_state_dict(torch.load(model_file, map_location=device))
        net.eval()
        self.model = net
        
        # transform
        mean = np.array([[[123.675, 116.28, 103.53]]], dtype=np.float32)
        std = np.array([[[58.395, 57.12, 57.375]]], dtype=np.float32)
        self.trans_img = [Resize(im_size=(512, 512)), Normalize(-1, mean, std, True)]

    def forward(self, cv2_img):
        # default inference batchsize 1
        ori_h, ori_w = cv2_img.shape[:2] 
        img_tensor = self.preprocess(cv2_img)
        with torch.no_grad():
            pred = self.model(img_tensor)
            pred = pred.sigmoid().detach().cpu().numpy()
        mask = self.postprocess(pred, ori_h, ori_w)
        return mask    

    def preprocess(self, cv2_img):
        # data aug
        cv2_img = cv2_img.astype(np.float32)
        for trans in self.trans_img:
            cv2_img = trans(cv2_img)
        img_data = torch.Tensor(cv2_img).float().permute(2, 0, 1).unsqueeze(0)
        if self.with_cuda:
            img_data = img_data.cuda()
        return img_data


    def postprocess(self, pred, ori_h, ori_w):
        '''
        pred: np.array [1x1xhxw]
        '''
        # gray to rgb
        pred = np.squeeze(pred).astype(np.float32)
        # resize
        if pred.shape[0] != ori_h or pred.shape[1] != ori_w:
            pred = cv2.resize(pred, dsize=(ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        pred = np.where(pred > self.thresh, 255., 0.)
        pred = np.stack([pred for _ in range(3)], axis=2).astype(np.uint8)
        # resize
        #pred = cv2.resize(pred, dsize=(ori_w, ori_h))
        return pred

