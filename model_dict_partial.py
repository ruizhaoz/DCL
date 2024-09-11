import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.distributions import Categorical
from wrn import Wide_ResNet

from ShuffleNetv2 import ShuffleV2
from mobilenetv2 import mobile_half
import torchvision.models as models
from small_resnet import ResNet10_xxxs, ResNet10_xxs, ResNet10_xs, ResNet10_s, ResNet10_m, ResNet10_l, ResNet10


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NewResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, all_planes=[64, 128, 256, 512], adaptive_pool=False):
        super(NewResNet, self).__init__()
        print('num-classes ', num_classes)
        self.in_planes = all_planes[0]  # 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, all_planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, all_planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, all_planes[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(all_planes[3] * block.expansion, num_classes)
        self.teacher = nn.Linear(all_planes[3] * block.expansion, num_classes)

        self.adaptive_pool = adaptive_pool
        if self.adaptive_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_channels = [all_planes[2]]
        self.xchannels = [all_planes[3] * block.expansion]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_message(self):
        return 'EfficientNetV2_s (CIFAR)'  # self.message

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        ft = out

        out = self.layer4(out)

        if self.adaptive_pool:
            out = self.avg_pool(out)
        else:
            out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)
        features = out

        out = self.linear(out)
        logits = out
        # return out
        return features, logits, ft  # feature of last layer (flattened), logits,


class PResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, all_planes=[64, 128, 256, 512], adaptive_pool=False):
        super(PResNet, self).__init__()
        print('num-classes ', num_classes)
        self.in_planes = all_planes[2]  # 64

        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        # self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, all_planes[1], num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, all_planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, all_planes[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(all_planes[3] * block.expansion, num_classes)

        self.adaptive_pool = adaptive_pool
        if self.adaptive_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_channels = [all_planes[2]]
        self.xchannels = [all_planes[3] * block.expansion]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_message(self):
        return 'EfficientNetV2_s (CIFAR)'  # self.message

    def forward(self, f):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        #
        # ft = [out]

        out = self.layer4(f)

        if self.adaptive_pool:
            out = self.avg_pool(out)
        else:
            out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)
        features = out

        out = self.linear(out)
        logits = out
        # return out
        return features, logits, f  # feature of last layer (flattened), logits,

class P2ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, all_planes=[64, 128, 256, 512], adaptive_pool=False):
        super(P2ResNet, self).__init__()
        print('num-classes ', num_classes)
        self.in_planes = all_planes[1]  # 64

        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        # self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, all_planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, all_planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, all_planes[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(all_planes[3] * block.expansion, num_classes)

        self.adaptive_pool = adaptive_pool
        if self.adaptive_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_channels = [all_planes[2]]
        self.xchannels = [all_planes[3] * block.expansion]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_message(self):
        return 'EfficientNetV2_s (CIFAR)'  # self.message

    def forward(self, f):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        #
        # ft = [out]

        out = self.layer4(f)

        if self.adaptive_pool:
            out = self.avg_pool(out)
        else:
            out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)
        features = out

        out = self.linear(out)
        logits = out
        # return out
        return features, logits, f  # feature of last layer (flattened), logits,

def ResNet18(num_classes=10, adaptive_pool=False):
    return NewResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, adaptive_pool=adaptive_pool)

def PResNet18(num_classes=10, adaptive_pool=False):
    return PResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, adaptive_pool=adaptive_pool)


def ResNet34(num_classes=10, adaptive_pool=False):
    return NewResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, adaptive_pool=adaptive_pool)


def ResNet50(num_classes=10, adaptive_pool=False):
    return NewResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, adaptive_pool=adaptive_pool)


# def Resnet8x4(**kwargs):
#     return ResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)


def get_model_from_name(num_classes, name, dataset):
    original_resnet = set(['ResNet18', 'ResNet34', 'ResNet50', ])
    if dataset == 'aug-tiny-imagenet-200' or dataset == 'imagenet-1k':
        adaptive_pool = True
    else:
        adaptive_pool = True

    if name == 'ResNet18':
        model = ResNet18(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'PResNet18':
        model = PResNet18(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'ResNet34':
        model = ResNet34(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'ResNet50':
        model = ResNet50(num_classes, adaptive_pool=adaptive_pool)

    elif name == 'ResNet10':
        model = ResNet10(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'ResNet10_l':
        model = ResNet10_l(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'ResNet10_m':
        model = ResNet10_m(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'ResNet10_s':
        model = ResNet10_s(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'ResNet10_xs':
        model = ResNet10_xs(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'ResNet10_xxs':
        model = ResNet10_xxs(num_classes, adaptive_pool=adaptive_pool)
    elif name == 'ResNet10_xxxs':
        model = ResNet10_xxxs(num_classes, adaptive_pool=adaptive_pool)


    elif name == 'ShuffleNetV2':
        model = ShuffleV2(num_classes=num_classes, adaptive_pool=adaptive_pool)
    elif name == 'MobileNetV2':
        model = mobile_half(num_classes)
    elif name == 'efficientnet_b0':  # this is for fine-tunning experiments
        model = models.efficientnet_b0(pretrained=True)  # this is for fine-tunning experiments
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    elif name == 'WRN28-2':  # this is for fine-tunning experiments
        model = Wide_ResNet(28, 2, 0.3, num_classes)
    elif name == 'WRN28-10':  # this is for fine-tunning experiments
        model = Wide_ResNet(28, 10, 0.3, num_classes)


    elif name == 'efficientnet_b7':  # this is for fine-tunning experiments
        model = models.efficientnet_b7(pretrained=True)  # this is for fine-tunning experiments
        model.classifier[1] = nn.Linear(in_features=2560, out_features=num_classes)
    else:
        assert ('Model not defined.')

    return model


def test(name):
    import numpy as np
    dataset = 'CIFAR-100'
    dataset = 'aug-tiny-imagenet-200'
    if dataset == 'CIFAR-100':
        num_classes = 100
        xshape = (1, 3, 32, 32)
    elif dataset == 'aug-tiny-imagenet-200':
        num_classes = 200
        xshape = (1, 3, 64, 64)
    # net = ResNet50(num_classes)
    # net = ResNet34(num_classes)
    # net = ResNet18(num_classes)
    # net = get_model_from_name( num_classes, name, 'CIFAR-100' )
    net = get_model_from_name(num_classes, name, dataset)

    from utils import get_model_infos

    flop, param = get_model_infos(net, xshape)

    print(' Model -- ' + name)
    print(
        "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )
    print()

    counts = sum(np.prod(v.size()) for v in net.parameters())
    print(counts)


'''
test( 'ResNet18' )
test( 'ResNet50' )
#test( 'MobileNetV2' )
test( 'ShuffleNetV2' )
#'''
