import torch
import torch.nn as nn
import torch.nn.functional as F
from .Net_block import *

import numpy as np

dropout_rate = 0.5

class UNet(nn.Module):
    def __init__(self, training, inchannel, num_organ, group_factor=1, channel_group=[32, 64, 128, 256, 512], norm_group=[4, 4, 4, 4, 4]):
        super().__init__()

        self.training = training

        channel_group = [int(g / group_factor) for g in channel_group]
        norm_group = [int(g / group_factor) for g in norm_group]

        self.encoder_stage1 = Conv(inchannel, channel_group[0], norm_group[0])

        self.encoder_stage1_2 = Conv(inchannel + num_organ + 1, channel_group[0], norm_group[0])

        self.encoder_stage2 = Conv(channel_group[1], channel_group[1], norm_group[1])
        self.encoder_stage3 = Conv(channel_group[2], channel_group[2], norm_group[2])
        self.encoder_stage4 = Conv(channel_group[3], channel_group[3], norm_group[3])
        self.encoder_stage5 = Conv(channel_group[4], channel_group[4], norm_group[4])

        self.decoder_stage1 = Conv(2*channel_group[3], channel_group[3], norm_group[3])
        self.decoder_stage2 = Conv(2*channel_group[2], channel_group[2], norm_group[2])
        self.decoder_stage3 = Conv(2*channel_group[1], channel_group[1], norm_group[1])
        self.decoder_stage4 = Conv(2*channel_group[0], channel_group[0], norm_group[0])

        # resolution reduce by half, and double the channels
        self.down_conv1 = Down(channel_group[0], channel_group[1])
        self.down_conv2 = Down(channel_group[1], channel_group[2])
        self.down_conv3 = Down(channel_group[2], channel_group[3])
        self.down_conv4 = Down(channel_group[3], channel_group[4])

        # double the resolution and reduce the channels
        self.up_conv1 = Up(channel_group[4], channel_group[3])
        self.up_conv2 = Up(channel_group[3], channel_group[2])
        self.up_conv3 = Up(channel_group[2], channel_group[1])
        self.up_conv4 = Up(channel_group[1], channel_group[0])

        self.map1 = nn.Conv2d(channel_group[3], num_organ + 1, 1)
        self.map2 = nn.Conv2d(channel_group[2], num_organ + 1, 1)
        self.map3 = nn.Conv2d(channel_group[1], num_organ + 1, 1)
        self.map4 = nn.Conv2d(channel_group[0], num_organ + 1, 1)


    def forward(self, inputs, stage=1):

        if stage > 1:
            long_range1 = self.encoder_stage1_2(inputs)
        else:
            long_range1 = self.encoder_stage1(inputs)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1)
        long_range2 = F.dropout(long_range2, dropout_rate, self.training)
        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2)
        long_range3 = F.dropout(long_range3, dropout_rate, self.training)
        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3)
        long_range4 = F.dropout(long_range4, dropout_rate, self.training)
        short_range4 = self.down_conv4(long_range4)

        outputs_0 = self.encoder_stage5(short_range4)
        outputs = F.dropout(outputs_0, dropout_rate, self.training)

        # print('-------decoder-------')
        short_range5 = self.up_conv1(outputs)
        outputs_1 = self.decoder_stage1(torch.cat([short_range5, long_range4], dim=1))
        outputs = F.dropout(outputs_1, dropout_rate, self.training)

        short_range6 = self.up_conv2(outputs)
        outputs_2 = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1))

        outputs = F.dropout(outputs_2, dropout_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs_3 = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1))

        outputs = F.dropout(outputs_3, dropout_rate, self.training)

        short_range8 = self.up_conv4(outputs)
        # print('self.up_conv4 = ', short_range8.shape)

        outputs_4 = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))

        if not self.training and stage == 2:
            outputs = self.map4(outputs_4)
        else:
            outputs = [self.map1(outputs_1), self.map2(outputs_2), self.map3(outputs_3), self.map4(outputs_4)]

        return outputs

class Net(nn.Module):
    def __init__(self, training, num_organ):
        super().__init__()

        self.training = training

        self.stage1 = UNet(training=training, inchannel=1, num_organ=num_organ)

    def forward(self, inputs, stage=2):

        output_stage1 = self.stage1(inputs)

        input_stage2 = F.softmax(output_stage1[-1], dim=1)
        input_stage2 = torch.cat((inputs, input_stage2), dim=1)
        output_stage2 = self.stage1(input_stage2, stage=stage)

        if self.training:
            return output_stage1, output_stage2
        else:
            return output_stage2

    def weight_init(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight.data, 0.25)
            nn.init.constant_(module.bias.data, 0)

