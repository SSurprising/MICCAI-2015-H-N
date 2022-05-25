import torch
import torch.nn as nn
import torch.nn.functional as F
from .Net_block import *

import numpy as np

dropout_rate = 0.5

class UNet(nn.Module):
    def __init__(self, training, inchannel, num_organ, group_factor=1, channel_group=[8, 16, 32, 64], norm_group=[2, 2, 2, 2], dimension=2):
        super().__init__()

        self.training = training

        channel_group = [int(g / group_factor) for g in channel_group]
        norm_group = [int(g / group_factor) for g in norm_group]

        self.encoder_stage1 = Conv(inchannel, channel_group[0], norm_group[0], dimension=dimension)
        self.encoder_stage2 = Conv(channel_group[1], channel_group[1], norm_group[1], dimension=dimension)
        self.encoder_stage3 = Conv(channel_group[2], channel_group[2], norm_group[2], dimension=dimension)
        self.encoder_stage4 = Conv(channel_group[3], channel_group[3], norm_group[3], dimension=dimension)

        self.decoder_stage1 = Conv(2*channel_group[-2], channel_group[-2], norm_group[-1], dimension=dimension)
        self.decoder_stage2 = Conv(2*channel_group[-3], channel_group[-3], norm_group[-2], dimension=dimension)
        self.decoder_stage3 = Conv(2*channel_group[-4], channel_group[-4], norm_group[-3], dimension=dimension)

        # resolution reduce by half, and double the channels
        self.down_conv1 = Down(channel_group[0], channel_group[1], dimension=dimension)
        self.down_conv2 = Down(channel_group[1], channel_group[2], dimension=dimension)
        self.down_conv3 = Down(channel_group[2], channel_group[3], dimension=dimension)

        # double the resolution and reduce the channels
        self.up_conv1 = Up(channel_group[-1], channel_group[-2], dimension=dimension)
        self.up_conv2 = Up(channel_group[-2], channel_group[-3], dimension=dimension)
        self.up_conv3 = Up(channel_group[-3], channel_group[-4], dimension=dimension)

        if dimension == 2:
            self.map = nn.Conv2d(channel_group[0], num_organ + 1, 1)
        elif dimension == 3:
            self.map = nn.Conv3d(channel_group[0], num_organ + 1, 1)

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)
        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1)
        long_range2 = F.dropout(long_range2, dropout_rate, self.training)
        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2)
        long_range3 = F.dropout(long_range3, dropout_rate, self.training)
        short_range3 = self.down_conv3(long_range3)

        outputs_0 = self.encoder_stage4(short_range3)
        outputs = F.dropout(outputs_0, dropout_rate, self.training)

        # print('-------decoder-------')
        short_range4 = self.up_conv1(outputs)
        outputs_1 = self.decoder_stage1(torch.cat([short_range4, long_range3], dim=1))
        outputs = F.dropout(outputs_1, dropout_rate, self.training)

        short_range5 = self.up_conv2(outputs)
        outputs_2 = self.decoder_stage2(torch.cat([short_range5, long_range2], dim=1))
        outputs = F.dropout(outputs_2, dropout_rate, self.training)

        short_range6 = self.up_conv3(outputs)
        outputs_3 = self.decoder_stage3(torch.cat([short_range6, long_range1], dim=1))

        outputs = self.map(outputs_3)
        # print('self.map = ', outputs_without_ensemble.shape)

        return outputs

class Net(nn.Module):
    def __init__(self, training, num_organ, dimension=2):
        super().__init__()

        self.training = training

        self.stage1 = UNet(training=training, inchannel=1, num_organ=num_organ, dimension=dimension)

    def forward(self, inputs):
        output_stage1 = self.stage1(inputs)

        return output_stage1

    def weight_init(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight.data, 0.25)
            nn.init.constant_(module.bias.data, 0)

