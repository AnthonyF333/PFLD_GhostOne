#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import torch
from torch.nn import Module, AvgPool2d, Linear
from models.base_module import MobileOneBlock, GhostOneBottleneck, Conv_Block


class PFLD_GhostOne(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98, inference_mode=False):
        super(PFLD_GhostOne, self).__init__()

        self.inference_mode = inference_mode
        self.num_conv_branches = 6

        self.conv1 = MobileOneBlock(in_channels=3,
                                    out_channels=int(64 * width_factor),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    inference_mode=self.inference_mode,
                                    use_se=False,
                                    num_conv_branches=self.num_conv_branches,
                                    is_linear=False)
        self.conv2 = MobileOneBlock(in_channels=int(64 * width_factor),
                                    out_channels=int(64 * width_factor),
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=int(64 * width_factor),
                                    inference_mode=self.inference_mode,
                                    use_se=False,
                                    num_conv_branches=self.num_conv_branches,
                                    is_linear=False)

        self.conv3_1 = GhostOneBottleneck(int(64 * width_factor), int(96 * width_factor), int(80 * width_factor), stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv3_2 = GhostOneBottleneck(int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv3_3 = GhostOneBottleneck(int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)

        self.conv4_1 = GhostOneBottleneck(int(80 * width_factor), int(200 * width_factor), int(96 * width_factor), stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv4_2 = GhostOneBottleneck(int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv4_3 = GhostOneBottleneck(int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)

        self.conv5_1 = GhostOneBottleneck(int(96 * width_factor), int(336 * width_factor), int(144 * width_factor), stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv5_2 = GhostOneBottleneck(int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv5_3 = GhostOneBottleneck(int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv5_4 = GhostOneBottleneck(int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)

        self.conv6 = GhostOneBottleneck(int(144 * width_factor), int(216 * width_factor), int(16 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv7 = MobileOneBlock(in_channels=int(16 * width_factor),
                                    out_channels=int(32 * width_factor),
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    inference_mode=self.inference_mode,
                                    use_se=False,
                                    num_conv_branches=self.num_conv_branches,
                                    is_linear=False)
        self.conv8 = Conv_Block(int(32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)

        self.avg_pool1 = AvgPool2d(input_size // 2)
        self.avg_pool2 = AvgPool2d(input_size // 4)
        self.avg_pool3 = AvgPool2d(input_size // 8)
        self.avg_pool4 = AvgPool2d(input_size // 16)

        self.fc = Linear(int(512 * width_factor), landmark_number * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x3 = self.avg_pool3(x)
        x3 = x3.view(x3.size(0), -1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x4 = self.avg_pool4(x)
        x4 = x4.view(x4.size(0), -1)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)
        x5 = x5.view(x5.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)
        landmarks = self.fc(multi_scale)

        return landmarks
