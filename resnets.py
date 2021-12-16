import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


# def _weights_init(m):
#     classname = m.__class__.__name__
#     # print(classname)
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BaseResidualBlock(nn.Module):
    expansion_factor = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1)):  # , option='A'):
        super(BaseResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.down_sample = None
        if stride != 1 or in_channels != out_channels:
            self.down_sample = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2],
                                                           (0, 0, 0, 0, out_channels // 4, out_channels // 4),
                                                           "constant",
                                                           0))

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.down_sample is not None:
            identity = self.down_sample(identity)

        x += identity
        x = F.relu(x, inplace=True)
        return x


class ResNet(nn.Module):
    def __init__(self, res_block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._initiate_layer(res_block, 16, num_blocks[0], stride=1)
        self.layer2 = self._initiate_layer(res_block, 32, num_blocks[1], stride=2)
        self.layer3 = self._initiate_layer(res_block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        # self.apply(_weights_init)

    def _initiate_layer(self, res_block, out_channels, num_blocks, stride):
        strides = [(stride, stride)]
        strides.extend([(1, 1)] * (num_blocks - 1))
        layers = []
        for stride in strides:
            layers.append(res_block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * res_block.expansion_factor

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet20():
    return ResNet(BaseResidualBlock, [3, 3, 3])


def resnet32():
    return ResNet(BaseResidualBlock, [5, 5, 5])


def resnet44():
    return ResNet(BaseResidualBlock, [7, 7, 7])


def resnet56():
    return ResNet(BaseResidualBlock, [9, 9, 9])


def resnet110():
    return ResNet(BaseResidualBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BaseResidualBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
