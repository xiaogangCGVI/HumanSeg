import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet34, resnet18

class LargeSeparableConv(nn.Module):

    def __init__(self, chns_in, chns_out, chns_mid, ksize):
        super(LargeSeparableConv, self).__init__()
        pad_size = int(ksize/2)
        self.col_max = nn.Conv2d(chns_in, chns_mid, (ksize,1), padding=(pad_size,0))
        self.col = nn.Conv2d(chns_mid, chns_out, (1,ksize), padding=(0,pad_size))
        self.row_max = nn.Conv2d(chns_in, chns_mid, (1,ksize), padding=(0,pad_size))
        self.row = nn.Conv2d(chns_mid, chns_out, (ksize,1), padding=(pad_size,0))

    def forward(self, x):
        col_max = self.col_max(x)
        col = self.col(col_max)
        row_max = self.row_max(x)
        row = self.row(row_max)
        return F.relu(col + row, inplace=True)


class RefineNet(nn.Module):
    
    def __init__(self, backend, num_output, conv_size, fpn_dim=128,
            scale_predict=1, depth_wise_conv=False, large_conv=-1):
        super(RefineNet, self).__init__()
        self.conv_size = conv_size
        self.backend = backend
        self.scale_predict = scale_predict

        # adaptive 
        self.adaptive = []
        for in_chns in conv_size:
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)

        # output conv
        self.smooth = []
        for i in range(len(conv_size)-1):
            if large_conv>0:
                smooth_layer = LargeSeparableConv(fpn_dim, fpn_dim, int(fpn_dim/2), ksize=large_conv)
            else:
                smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, bias=True, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)
        self.predict = nn.Conv2d(fpn_dim, num_output, kernel_size=1, bias=True)


    def forward(self, x):
        if self.backend is not None:
            conv_ftr = self.backend(x)
        else:
            conv_ftr = x

        h = [ adaptive_layer(conv) for adaptive_layer, conv in zip(self.adaptive, conv_ftr) ]
        
        f = h[-1]
        for smooth_layer, fpn_in in reversed(list(zip(self.smooth, h))):
            f = F.interpolate(f, scale_factor=2, mode='nearest') + fpn_in
            f = smooth_layer(f)
        predict = self.predict(f)
        
        if self.scale_predict != 1:
            predict = F.interpolate(predict, scale_factor=self.scale_predict, mode='bilinear', align_corners=True)

        return predict

def get_network(name, pretrained):
    if name == 'refine34':
        backend = resnet34(pretrained=pretrained)
        net = RefineNet(backend=backend, num_output=1, conv_size=[64,128,256,512], fpn_dim=64)
    elif name == 'refine18':
        backend = resnet18(pretrained=pretrained)
        net = RefineNet(backend=backend, num_output=1, conv_size=[64,128,256,512], fpn_dim=64)
    else:
        print('This network is not supported !')
        net = None
    return net

if __name__ == '__main__':

    # test net
    '''backend = resnet34(pretrained=True)
    print(backend)
    net = RefineNet(backend, 1, (64,128,256,512),scale_predict=1, fpn_dim=128)
    x = torch.rand(1, 3, 512, 512)
    y = net(x)
    print(y.size())'''

    # sync bn
    from sync_batchnorm import convert_model
    # m is a standard pytorch model
    model = get_network('refine18', pretrained=True)
    model = nn.DataParallel(model)
    # after convert, m is using SyncBN
    model = convert_model(model)
    print(type(model).__name__)
