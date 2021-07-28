import os
import sys
import cv2
import argparse
import numpy as np
from torch.utils.data.dataset import Dataset

from tqdm import tqdm
from loss import get_loss
from refinenet import get_network
from dataset import HumanDataset, MattingDataset
from trainer import Trainer
from inference import InferenceWrapper
from torch.utils.data import ConcatDataset
from utils import draw_mask, generate_mask_path, mean_iou


def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Mask R-CNN Demo')
    parser.add_argument('--mode', help='train or eval', default='train', type=str)
    parser.add_argument('--arch', help='model name', default='refine34', type=str)
    parser.add_argument('--batch_size', help='batch size', default=10, type=int)
    parser.add_argument('--num_workers', help='dataloder worker number', default=4, type=int)
    parser.add_argument('--anno_file', help='annotation list', default=[], type=str, nargs='+')
    parser.add_argument('--thresh', help='segmentation threshold', default=0.5, type=float)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--lr_step', help="learning schedule", default=[20, 30], type=int, nargs='+')
    parser.add_argument('--epochs', help='training epochs', default=40, type=int)
    parser.add_argument('--save_step', help='save model each save step', default=3, type=int)
    parser.add_argument('--save_path', help="outdir", default="./output", type=str)
    parser.add_argument('--save_tag', help="tag", default="default", type=str)
    parser.add_argument('--model_file', help="outdir", default=None, type=str)
    parser.add_argument('--clip_grad', help="clip grad during training", default=None, type=float)
    parser.add_argument('--gpu_num', help="the number of gpu", default=1, type=int)
    parser.add_argument('--vis_path', help="vis save path", default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def train(args, criterion):

    # dataloader informations
    dataset = ConcatDataset([HumanDataset(args.anno_file[0], is_train=True), 
                             MattingDataset(args.anno_file[1], is_train=True)])
    dataloader_info = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'drop_last': True,
    }
    # auto pretrained during training
    model = get_network(args.arch, True)

    # training process
    epochs = args.epochs
    save_step = args.save_step
    tensorboard_dir = os.path.join(args.save_path, args.save_tag)
    trainer = Trainer(model, dataloader_info, criterion, args.lr, args.lr_step, args.gpu_num, 
                      scale_ratio=0.25, iter_size=1, outdir=args.save_path, tensorboard_dir=tensorboard_dir, grad_clip=args.clip_grad)
    
    for epoch_id in range(1, epochs+1):
        # train
        trainer.lr_adjust(epoch_id)
        trainer.train_one_epoch(epoch_id)
        #trainer.lr_adjust(epoch_id)
        
        # checkpoint
        if epoch_id % save_step == 0:
            trainer.save_model(epoch_id=epoch_id, tag=args.save_tag)


def test(args):
    # tester
    model = get_network(args.arch, True)
    tester = InferenceWrapper(model, args.model_file, args.gpu_num>0, thresh=args.thresh)

    # load anno list
    f = open(args.anno_file[0], 'r')
    img_paths = [line.strip() for line in f]
    mask_paths = [generate_mask_path(p, True, 'matting') for p in img_paths]
    vis_path = args.vis_path
    iou = 0.
    # background
    bk = cv2.imread('./background/star.jpeg')
    bk = cv2.resize(bk, (600, 800))
    for i in tqdm(range(len(img_paths))):
        #import pdb;pdb.set_trace()
        img = cv2.imread(img_paths[i])
        mask = cv2.imread(mask_paths[i], cv2.IMREAD_UNCHANGED)[:,:,3]
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = np.stack([mask, mask, mask], axis=2).astype(np.uint8)
        try:
            pred = tester.forward(img)
        except:
            continue
        
        # vis
        if vis_path is not None:
            img_name = os.path.basename(img_paths[i])
            save_path = os.path.join(vis_path, img_name)
            #cv2.imshow('demo', np.concatenate([img, mask, pred], axis=1))
            #cv2.waitKey()
            pred_mask = pred[:,:,0:1] / 255.
            bk_img = pred_mask * img + (1. - pred_mask) * bk
            cv2.imwrite(save_path, np.concatenate([img, bk_img.astype(np.uint8), pred], axis=1))
        iou += mean_iou(pred, mask)
        #import pdb;pdb.set_trace()

    print('Mean Iou: {}'.format(iou / len(img_paths)))


if __name__ == '__main__':
    args = parse_args()
    # train
    if args.mode == 'train':
        criterion = get_loss('bce')
        #criterion = get_loss('focal')
        train(args, criterion)
    
    # test
    if args.mode == 'eval':
        test(args)

