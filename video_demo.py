import threading
import time
import cv2
import numpy as np

from utils import draw_mask
from refinenet import get_network
from inference import InferenceWrapper

def img_to_mask(cv2_img, tester):
    mask = tester.forward(cv2_img)
    vis_img = draw_mask(cv2_img, mask)
    return vis_img

def initial_tester(model_file, gpu_num=0, thresh=0.5):
    # tester
    model = get_network('refine18', True)
    tester = InferenceWrapper(model, model_file, gpu_num>0, thresh=thresh)
    return tester

if __name__ == '__main__':
    # video
    algo_tester = initial_tester('/Users/tiangang.zhang/mnt/home/toc/SDE/HumanSeg/output/bce_sync_bn_matting/epoch_009.pth')
    #import pdb;pdb.set_trace()
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        img = img_to_mask(frame, algo_tester)

        # Display the resulting frame
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

