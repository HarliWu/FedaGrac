# -*- coding: utf-8 -*-

'''
This is full set for cifar datasets (CIFAR-10 and CIFAR100) 
Models: LR, ResNet, VGG, AlexNet
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

def get_transform():
    train_transform = transforms.Compose([ 
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])
    test_transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])
    return [train_transform, test_transform]

def load_cifar_datasets(path='./dataset', n_class=10, train_transform=get_transform()[0], test_transform=get_transform()[1]):
    if n_class == 10:
        train_dataset = CIFAR10(path, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(path, train=False, download=True, transform=test_transform)
    elif n_class == 100:
        train_dataset = CIFAR100(path, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(path, train=False, download=True, transform=test_transform)
    else:
        train_dataset, test_dataset = None, None
    return train_dataset, test_dataset


############# Logistic Regression #############

class LogisticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LogisticRegression,self).__init__()
        self.logistic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        x=x.view(x.size(0), -1).contiguous()
        out = self.logistic(x)
        # out = F.log_softmax(out, dim=1)
        return out

def LR(n_class):
    return LogisticRegression(3072,n_class)

###############################################



############# ResNet #############

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_class=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, n_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(n_class=10):
    return ResNet(BasicBlock, [2,2,2,2], n_class=10)

def ResNet34(n_class=10):
    return ResNet(BasicBlock, [3,4,6,3], n_class=10)

def ResNet50(n_class=10):
    return ResNet(Bottleneck, [3,4,6,3], n_class=10)

def ResNet101(n_class=10):
    return ResNet(Bottleneck, [3,4,23,3], n_class=10)

def ResNet152(n_class=10):
    return ResNet(Bottleneck, [3,8,36,3], n_class=10)

###############################################



############# VGG #############

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, n_class=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, n_class),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(n_class=10):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), n_class=10)


def vgg11_bn(n_class=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), n_class=10)


def vgg13(n_class=10):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), n_class=10)


def vgg13_bn(n_class=10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), n_class=10)


def vgg16(n_class=10):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), n_class=10)


def vgg16_bn(n_class=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), n_class=10)


def vgg19(n_class=10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), n_class=10)


def vgg19_bn(n_class=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), n_class=10)

###############################################



############# AlexNet #############

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048), 
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # return F.log_softmax(x, dim=1)
        return x

###############################################


##################################

if __name__ == "__main__":
    import torchsummary
    # print("ResNet8")
    model = AlexNet()
    torchsummary.summary(model, (3, 32, 32), depth=5, verbose=1)
    # print("\n\n\n")
    # model = ResNet8(0, False, 200)
    # torchsummary.summary(model, (3, 64, 64), depth=5, verbose=1)