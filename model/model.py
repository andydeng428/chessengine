# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.num_planes = 12 + 5  # 12 piece planes + 5 scalar features converted into planes
        self.num_filters = 256
        self.num_res_blocks = 39

        # Initial convolutional layer
        self.conv = nn.Conv2d(self.num_planes, self.num_filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.num_filters)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.num_filters) for _ in range(self.num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(self.num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4208)  #Match the move encoding

        # Value head
        self.value_conv = nn.Conv2d(self.num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))

        # Residual blocks
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        policy = self.policy_relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        # policy = F.softmax(policy, dim=1)
        # when training, dont use softmax, when converting to onnx we use softmax

        # Value head
        value = self.value_relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
