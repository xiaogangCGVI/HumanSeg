from PIL import Image
from tqdm import tqdm
from utils import generate_mask_path
import cv2

# supervisely
'''
l = '/data/SDE/dataset/SuperviselyPersonDataset/train.txt'
f = open(l)
paths = [line.strip()  for line in f]
for p in paths:
    img = Image.open(p)
    if len(img.split()) < 3:
        print(p)
'''


# matting
l = '/data/SDE/dataset/human_half/train.txt'
f = open(l)
paths = [line.strip() for line in f]
for p in tqdm(paths):
    m = generate_mask_path(p, True, 'matting')
    mask = cv2.imread(m, cv2.IMREAD_UNCHANGED)
    try:
        mask = mask[:,:,3] / 255.
    except:
        print(p) 
