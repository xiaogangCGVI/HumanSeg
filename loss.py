from torch.nn import BCELoss
import torch.nn.functional as F


def bce_loss(img_data, target, scale_ratio=1, with_cuda=True):
    loss_layer = BCELoss(reduction='mean')
    if with_cuda:
        loss_layer = loss_layer.cuda()
    # process data tensor and target
    if scale_ratio != 1:
        h, w = target.size()[-2:]
        size = (int(h*scale_ratio), int(w*scale_ratio))
        #target = F.interpolate(target, scale_factor=scale_ratio, mode='nearest')
        target = F.interpolate(target, size=size, mode='nearest')
    img_data = img_data.sigmoid()
    # cal loss
    loss = loss_layer(img_data, target)
    return loss


def focal_loss(img_data, target, scale_ratio=1, with_cuda=True, gamma=2.0, alpha=0.75):
    loss_layer = BCELoss(reduction='none')
    if with_cuda:
        loss_layer = loss_layer.cuda()
    if scale_ratio != 1:
        h, w = target.size()[-2:]
        size = (int(h*scale_ratio), int(w*scale_ratio))
        #target = F.interpolate(target, scale_factor=scale_ratio, mode='nearest')
        target = F.interpolate(target, size=size, mode='nearest')
    pred = img_data.sigmoid()
    # cal focal weight
    pt = (1-pred) * target + pred * (1-target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    # cal loss
    loss = loss_layer(pred, target) * weight
    loss = loss.mean()
    return loss


def get_loss(name):
    if name == 'bce':
        return bce_loss
    elif name == 'focal':
        return focal_loss
    else:
        print('This Loss is not supported !')
        return  None
    

