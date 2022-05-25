import torch
import torch.nn as nn
import torch.nn.functional as F
 
import numpy as np
 
# First stage
class ResUNet_1(nn.Module):
    def __init__(self, training, inchannel, stage, num_organ, group=32, dropout_rate=0.5):
        super().__init__()
 
        self.training = training
        self.stage = stage
        self.dropout_rate = dropout_rate
 
        self.encoder_stage1 = nn.Sequential(
            nn.Conv2d(inchannel, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )
 
        self.encoder_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
 
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
            
            nn.GroupNorm(group, 32),
        )
 
        self.encoder_stage3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
 
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
 
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
            
            nn.GroupNorm(group, 64),
        )
 
        self.encoder_stage4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
 
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
            
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
            
            nn.GroupNorm(group, 128),
        )
 
        self.encoder_stage5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
 
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
 
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
            
            nn.GroupNorm(group, 256),
        )
 
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, padding=1),
            nn.PReLU(128),
 
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
 
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
            
            nn.GroupNorm(group, 128),
        )
 
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, padding=1),
            nn.PReLU(64),
 
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
 
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
            
            nn.GroupNorm(group, 64),
        )
 
        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, padding=1),
            nn.PReLU(32),
 
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
            
            nn.GroupNorm(group, 32),
        )
 
        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, 1, padding=1),
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

        self.map = nn.Conv2d(16, num_organ + 1, 1)


            
    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs) + inputs
 
        short_range1 = self.down_conv1(long_range1)
 
        long_range2 = self.encoder_stage2(short_range1) + short_range1
 
        long_range2 = F.dropout(long_range2, self.dropout_rate, self.training)
 
        short_range2 = self.down_conv2(long_range2)
 
        long_range3 = self.encoder_stage3(short_range2) + short_range2
 
        long_range3 = F.dropout(long_range3, self.dropout_rate, self.training)
 
        short_range3 = self.down_conv3(long_range3)
 
        long_range4 = self.encoder_stage4(short_range3) + short_range3
 
        long_range4 = F.dropout(long_range4, self.dropout_rate, self.training)
 
        short_range4 = self.down_conv4(long_range4)
 
        outputs_0 = self.encoder_stage5(short_range4) + short_range4
        outputs = F.dropout(outputs_0, self.dropout_rate, self.training)
 
        # print('-------decoder-------')
 
        short_range5 = self.up_conv1(outputs)
        outputs_1 = self.decoder_stage1(torch.cat([short_range5, long_range4], dim=1)) + short_range5
        outputs = F.dropout(outputs_1, self.dropout_rate, self.training)
 
        short_range6 = self.up_conv2(outputs)
        outputs_2 = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
 
        outputs = F.dropout(outputs_2, self.dropout_rate, self.training)
 
        short_range7 = self.up_conv3(outputs)
 
        outputs_3 = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
 
        outputs = F.dropout(outputs_3, self.dropout_rate, self.training)
 
        short_range8 = self.up_conv4(outputs)
        # print('self.up_conv4 = ', short_range8.shape)
 
        outputs_4 = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
 
        outputs = self.map(outputs_4)
        # print('self.map = ', outputs_without_ensemble.shape)
 
        return outputs, (long_range1, long_range2, long_range3, long_range4), (
        outputs_4, outputs_3, outputs_2, outputs_1, outputs_0)

# second stage
class ResUNet_2(nn.Module):
    def __init__(self, training, inchannel, stage, num_organ, group=32, dropout_rate=0.5):
        super().__init__()
 
        self.training = training
        self.stage = stage
        self.dropout_rate = dropout_rate
 
        self.encoder_stage1 = nn.Sequential(
            nn.Conv2d(inchannel + 16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )
 
        self.encoder_stage2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, padding=1),
            nn.PReLU(32),
 
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
            
            nn.GroupNorm(group, 32),
        )
 
        self.encoder_stage3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, padding=1),
            nn.PReLU(64),
 
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
 
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
            
            nn.GroupNorm(group, 64),
        )
 
        self.encoder_stage4 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, padding=1),
            nn.PReLU(128),
 
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
 
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
            
            nn.GroupNorm(group, 128),
        )
 
        self.encoder_stage5 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, 1, padding=1),
            nn.PReLU(256),
 
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
 
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
            
            nn.GroupNorm(group, 256),
        )
 
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(128 + 128 + 128, 128, 3, 1, padding=1),
            nn.PReLU(128),
 
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
 
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
            
            nn.GroupNorm(group, 128),
        )
 
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(64 + 64 + 64, 64, 3, 1, padding=1),
            nn.PReLU(64),
 
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
 
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
            
            nn.GroupNorm(group, 64),
        )
 
        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, padding=1),
            nn.PReLU(32),
 
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
            
            nn.GroupNorm(group, 32),
        )
 
        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(16 + 16 + 16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )
 
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
 
        self.down_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, 2),
            nn.PReLU(256)
        )
 
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
 
        self.map = nn.Conv2d(16, num_organ + 1, 1)
 
    def forward(self, inputs, long_ranges=None, outputs_ranges=None):
        long_range1 = self.encoder_stage1(torch.cat([inputs, long_ranges[0]], dim=1))
        short_range1 = self.down_conv1(long_range1)
 
        long_range2 = self.encoder_stage2(torch.cat([short_range1, long_ranges[1]], dim=1)) + short_range1
        long_range2 = F.dropout(long_range2, self.dropout_rate, self.training)
        short_range2 = self.down_conv2(long_range2)
 
        long_range3 = self.encoder_stage3(torch.cat([short_range2, long_ranges[2]], dim=1)) + short_range2
        long_range3 = F.dropout(long_range3, self.dropout_rate, self.training)
        short_range3 = self.down_conv3(long_range3)
 
        long_range4 = self.encoder_stage4(torch.cat([short_range3, long_ranges[3]], dim=1)) + short_range3
        long_range4 = F.dropout(long_range4, self.dropout_rate, self.training)
        short_range4 = self.down_conv4(long_range4)
 
        outputs = self.encoder_stage5(torch.cat([short_range4, outputs_ranges[4]], dim=1)) + short_range4
        outputs = F.dropout(outputs, self.dropout_rate, self.training)
 
#        print('-------decoder-------')
        short_range5 = self.up_conv1(outputs)
        outputs = self.decoder_stage1(torch.cat([short_range5, long_range4, outputs_ranges[3]], dim=1)) + short_range5
        outputs = F.dropout(outputs, self.dropout_rate, self.training)
 
        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3, outputs_ranges[2]], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.dropout_rate, self.training)
 
        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2, outputs_ranges[1]], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.dropout_rate, self.training)
 
        short_range8 = self.up_conv4(outputs)
        # print('self.up_conv4 = ', short_range8.shape)
 
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1, outputs_ranges[0]], dim=1)) + short_range8
 
        outputs = self.map(outputs)
        # print('self.map = ', outputs_without_ensemble.shape)
 
        # 返回概率图
        return outputs

class Net(nn.Module):
    def __init__(self, training, num_organ, stages=2):
        super().__init__()
 
        self.training = training

        self.stage1 = ResUNet_1(training=training, inchannel=1, stage='stage1', num_organ=num_organ)
        # num_organ + 2: '1' for background, '1' for initial input
        self.stage2 = ResUNet_2(training=training, inchannel=num_organ + 2, stage='stage2', num_organ=num_organ)
        self.stages = stages
 
    def forward(self, inputs):
 
        # 得到第一阶段的结果
        output_stage1, long_ranges, outputs_ranges = self.stage1(inputs)
        # after self.stage1(), outputs_stage1 = 1 * 1 * 48 * 128 * 128
 
        # 将第一阶段的结果与原始输入数据进行拼接作为第二阶段的输入
        inputs_stage2 = torch.cat((output_stage1, inputs), dim=1)
        # inputs_stage2 = 1 * 15 * 48 * 256 * 256
 
        # 得到第二阶段的结果
        output_stage2 = self.stage2(inputs_stage2, long_ranges, outputs_ranges)
        # output_stage2 = 1 * 14 * 48 * 256 * 256

        if self.training is True:
            return output_stage1, output_stage2
        else:
            return output_stage2

    def weight_init(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight.data, 0.25)
            nn.init.constant_(module.bias.data, 0)
