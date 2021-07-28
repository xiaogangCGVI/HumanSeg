import cv2
import os
import numpy as np

# general utils
def generate_mask_path(path, return_path=False, mode='supervisely'):
    path_dir = os.path.dirname(path)
    path_file = os.path.basename(path)

    if mode == 'supervisely':
        new_path_dir = path_dir.replace('/img', '/mask')
        p, suffix = os.path.splitext(path_file)
        new_path_file = p + '_mask' + suffix
    elif mode == 'matting':
        new_path_dir = path_dir.replace('/clip_img', '/matting').replace('clip_', 'matting_')
        p, suffix = os.path.splitext(path_file)
        new_path_file = p + '.png'
    else:
        print('{} mode is not defined !'.format(mode))
        return None

    if return_path:
        return os.path.join(new_path_dir, new_path_file)
    return new_path_dir, new_path_file


# visualize utils
def draw_mask(img, mask, color=(255, 255, 0), ratio=0.6):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    weight =  (mask > 0).astype(np.float32) * ratio
    weight = cv2.merge([weight, weight, weight])
    colormap = np.array(color).reshape(1,1,3)
    combined = colormap*weight + img.astype(np.float32)*(1-weight)
    return combined.astype(np.uint8)


# evaluation utils
def pixel_accuracy():
    pass

def mean_iou(mask, gt):
    mask = mask.astype(np.float32)[:,:,0]
    gt = gt.astype(np.float32)[:,:,0]
    # cal union and intersection
    union = np.logical_or(gt>0, mask>0).astype(np.int)
    intersect = np.logical_and(gt>0, mask>0).astype(np.int)
    return np.sum(intersect) / float(np.sum(union))


if __name__ == '__main__':
    '''
    p, s = generate_mask_path('/a/b/c/img/image.png')
    print(p, s)
    '''
    img = cv2.imread('/Users/tiangang.zhang/mnt/data/SDE/dataset/matting_human_half/matting/1803151818/matting_00000000/1803151818-00000124.png', cv2.IMREAD_UNCHANGED)
    mask = img[:,:,3]
    print(img.shape)
    print(np.sum(np.logical_and(mask>128, mask <255)))
    cv2.imshow('demo', mask)
    cv2.waitKey()