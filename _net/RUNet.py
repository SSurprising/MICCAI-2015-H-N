import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

dropout_rate = 0.5

class UNet(nn.Module):
    def __init__(self, training, inchannel, num_organ):
        super().__init__()

        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv2d(inchannel, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.encoder_stage5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        # resolution reduce by half, and double the channels
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        # keep the resolution and double the channels
        self.down_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, 2),
            nn.PReLU(256)
        )

        # double the resolution and reduce the channels
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.PReLU(16)
        )

        self.map = nn.Sequential(
            nn.Conv2d(16, num_organ + 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
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

        outputs = self.map(outputs_4)
        # print('self.map = ', outputs_without_ensemble.shape)

        return outputs

# 定义最终的级连3D FCN
class Net(nn.Module):
    def __init__(self, training, num_organ):
        super().__init__()

        self.training = training

        self.stage1 = UNet(training=training, inchannel=1, num_organ=num_organ)

    def forward(self, inputs):

        # 得到第一阶段的结果
        output_stage1 = self.stage1(inputs)

        return output_stage1

    def weight_init(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight.data, 0.25)
            nn.init.constant_(module.bias.data, 0)

