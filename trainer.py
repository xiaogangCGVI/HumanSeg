import math
import os

import torch
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sync_batchnorm import convert_model


class Trainer(object):
    
    def __init__(self, model, dataloader_info, criterion, lr, lr_step, gpu_num, 
                 iter_size=1, outdir='./output', tensorboard_dir='./run', scale_ratio=0.25, grad_clip=None):
        self.target_ratio = scale_ratio

        # TODO: check defaultCollate func
        self.dataloader = DataLoader(**dataloader_info)
        self.loss_fn = criterion

        # TODO: 1. GPU dp (not ddp) how to use sync bn
        #       2. replace dp by dp syncbn
        if gpu_num == 1:
            self.model = DataParallel(model, device_ids=list(range(gpu_num)))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            #self.loss_fn.to(device)
        elif gpu_num > 1:
            self.model = convert_model(DataParallel(model, device_ids=list(range(gpu_num))))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        else:
            self.model = model.to(torch.device('cpu'))
        self.gpu_num = gpu_num

        # optimizer
        self.optimizer = self.get_optimizer(lr)
        self.lr_step = lr_step
        self.iter_size = iter_size

        # grad clip
        self.grad_clip = grad_clip

        # save dir
        self.outdir = outdir

        # initialize tensorboard
        self.writer = SummaryWriter(tensorboard_dir)

    def train_one_epoch(self, epoch_id):
        writer = self.writer
        # net
        model = self.model
        model.train()
        # data iterator
        dataloader = self.dataloader
        max_iters = int(math.floor(len(dataloader.dataset) / dataloader.batch_size))
        for batch_id, data in enumerate(dataloader):
            # Compute prediction and loss
            img, label = data[0], data[1]
            loss = self.train_one_step(img, label, model, batch_id)
            
            # display
            if batch_id % 20 == 0:
                print('Epoch[{:0>3d}] Iter[{:d}\{:d}] Loss: {:.4f} Lr: {:f}'.format(
                    epoch_id, batch_id, max_iters, loss.item(), self.get_lr()
                ))

            # add scalar
            writer.add_scalar('Segmentation Train Loss', loss.item(), (epoch_id - 1)*max_iters+batch_id)


    def train_one_step(self, img_data, target, model, iter_id):
        target_ratio = self.target_ratio
        loss_fn = self.loss_fn
        gpu_num =self.gpu_num

        # cpu or gpu
        if gpu_num > 0:
            img_data = img_data.cuda()
            target = target.cuda()
        
        # forward
        pred = model(img_data)
        loss = loss_fn(pred, target, target_ratio, gpu_num>0)
        # backward each iter size
        loss.backward()
        if iter_id % self.iter_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        return loss


    def save_model(self, epoch_id, tag='default'):
        model_name = 'epoch_%03d.pth' % (epoch_id)
        model_dir = os.path.join(self.outdir, tag)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        save_path = os.path.join(self.outdir, tag, model_name)
        if type(self.model).__name__ == 'DataParallel':
            save_model = self.model.module
        elif type(self.model).__name__ == 'DataParallelWithCallback':
            save_model = self.model.module
        else:
            save_model = self.model
        torch.save(save_model.state_dict(), save_path)
        return save_path


    def lr_adjust(self, epoch_id):
        optimizer = self.optimizer
        if epoch_id not in self.lr_step:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        #optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        return optimizer

if __name__ == '__main__':
    from dataset import HumanDataset
    from refinenet import get_network
    from loss import bce_loss
    dataloader_info = {
        'dataset': HumanDataset('/Users/tiangang.zhang/work/data/SuperviselyPersonDataset/train.txt', is_train=True),
        'batch_size': 10,
        'shuffle': True,
        'num_workers': 4,
        'drop_last': True,
    }
    model = get_network('refine34', True)
    trainer = Trainer(model, dataloader_info, bce_loss, 1e-3, [20, 30], 0)
    trainer.train_one_epoch(0)

