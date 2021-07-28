import os
import cv2
import random
import numpy as np
import supervisely_lib as sly

from tqdm import tqdm
from glob import glob
from utils import generate_mask_path


def download_mask(root):
    project = sly.Project(root, sly.OpenMode.READ)
    print("Project name: ", project.name)
    print(project.datasets.keys())
    # traverse and save
    for dataset in project:
        print('DATASET NAME: {}'.format(dataset.name))
        for item_name in tqdm(dataset):
            ann_path = dataset.get_ann_path(item_name)
            img_path = dataset.get_img_path(item_name)
            anno = sly.Annotation.load_json_file(ann_path, project.meta)
            mask = np.zeros(anno.img_size + (3, ), dtype=np.uint8)
            anno.draw(mask, color=(255, 255, 255))
            # debug
            '''ori_img = cv2.imread(img_path)
            vis_img = np.concatenate((ori_img, mask), axis=1)
            cv2.imshow('demo', vis_img)
            cv2.waitKey()'''
            #save
            path_dir, path_file = generate_mask_path(img_path)
            if not os.path.isdir(path_dir):
                os.mkdir(path_dir)
            cv2.imwrite(os.path.join(path_dir, path_file), mask)
            
def random_split_train_val(dataset_root, out_dir):
    #image_paths = glob(os.path.join(dataset_root, r'ds*/img/*'))
    image_paths = glob(os.path.join(dataset_root, r'clip_img/*/*/*'))
    # shuffle
    random.shuffle(image_paths)
    train_set = image_paths[:-1000]
    val_set = image_paths[-1000:]
    # save
    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        for line in train_set:
            f.write(line+'\n')

    with open(os.path.join(out_dir, 'val.txt'), 'w') as f:
        for line in val_set:
            f.write(line+'\n')
    print('Write Done !')

if __name__ == '__main__':
    # download
    '''dataroot = '/Users/tiangang.zhang/work/data/SuperviselyPersonDataset/'
    download_mask(dataroot)'''

    # random split
    #dataroot = '/data/SDE/dataset/SuperviselyPersonDataset/'
    #outdir = '/data/SDE/dataset/SuperviselyPersonDataset/'
    dataroot = '/data/SDE/dataset/human_half/'
    outdir = '/data/SDE/dataset/human_half/'
    random_split_train_val(dataroot, outdir)



