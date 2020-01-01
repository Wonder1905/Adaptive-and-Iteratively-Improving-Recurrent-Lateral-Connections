'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlockDy(nn.Module):
    expansion = 1

    def __init__(self, num_loops,in_planes, planes, stride=1, option='A'):
        super(BasicBlockDy, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.bn1 = nn.InstanceNorm2d(planes,track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.num_loops =num_loops
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=(5,5),  stride=(2,2),padding = (2,2))
        self.maxpool = nn.MaxPool2d(kernel_size=(5,5),  stride=(2,2),padding = (2,2))         

        self.block3_conv1_create_dy_w = nn.Linear(32,64*9)

        
    def create_deltaw(self, x,alpha):
        avg = self.avgpool(x)
        max = self.maxpool(x)
        h_tag = torch.cat((avg,max),dim=1)
        h_bar = h_tag.view(h_tag.shape[0],64,-1)
        #out = self.dropout(out)
        delta_w = self.block3_conv1_create_dy_w(h_bar)
        return delta_w 

    def metaconv(self, x, w):
        '''
        Forward pass of a meta convolution layer.
        Note that we do not conv all batch with the same set of conv weights.
        The trick is to use group convolutions for convolving each input with its own set of conv weights.
        '''
        holdx = x
        holdw = w
        w = torch.reshape(w,(w.shape[0] * w.shape[1], w.shape[2], 3,3))

        x = x.view(1, x.size(1) * x.size(0), x.size(2), x.size(3))
        out = F.conv2d(x, w, None,stride=1, groups=holdx.size(0),padding=1)
        
        out = out.view(holdx.size(0), holdw.shape[1], holdx.shape[2], holdx.shape[3])
        return out
    def forward(self, x,w,alpha=0.0001):
        beta = 1
        for loop in range(self.num_loops): 
        
           if loop == 0:
              total_w = w
              out = self.metaconv(x,w)
           else:
              out = self.metaconv(x,total_w)              
           out = F.relu(self.bn1(out))
           out = self.bn2(self.conv2(out))
           div = beta ** loop
           alpha = alpha / div
           delta_w = self.create_deltaw(out,alpha)
           delta_w = alpha* self.tanh(delta_w)
           delta_w = delta_w.view(w.shape)
           total_w = total_w + delta_w

        out += self.shortcut(x)
        out = F.relu(out)
        return out,total_w

class layer3dy(nn.Module):
    def __init__(self,num_loops,planes, num_blocks=3, stride=1):
       super(layer3dy, self).__init__()
       self.num_loops =num_loops
       self.block1 = BasicBlock(32, planes,2)
       self.block2 = BasicBlock(64, planes,1)
       self.block3 = BasicBlockDy(num_loops,64, planes,1)
       for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x,block3_conv1_w,alpha=0.0001):
       x = self.block1(x)
       x = self.block2(x)
       x,w = self.block3(x,block3_conv1_w,alpha)
       return x

class ResNet(nn.Module):
    def __init__(self,weights,num_loops, block=BasicBlock, num_blocks = [3, 3, 3] , num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)        
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = layer3dy(num_loops,planes = 64,num_blocks = 3,stride = 2)
        self.linear = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.apply(_weights_init)
        try:
           dict = torch.load(weights)['state_dict']
        except:
           dict = torch.load(weights)
        try:
           self.w = nn.Parameter(dict['layer3.2.conv1.weight'],requires_grad=False)
        except:
            self.w = nn.Parameter(dict['module.layer3.2.conv1.weight'],requires_grad=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,alpha=0.0001):
        cut = x.shape[0]
        w = self.w.repeat(cut,1,1,1,1)
        #w = self.w.repeat[0:cut,:,:,:,:]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out,w,alpha)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out




