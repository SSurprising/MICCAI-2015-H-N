import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self, num_organ):
        super().__init__()
        self.num_organ = num_organ

    def forward(self, pred, target):

        N_total = 1
        for lenth in target.shape:
            N_total *= lenth
        N_total = target.shape[0] * target.shape[1] * target.shape[2]
        #
        w = torch.ones(self.num_organ + 1)
        labels, counts = target.unique(return_counts=True)
        # print('label', labels)
        # print('counts', counts)
        labels = [int(l) for l in labels]
        for i in range(len(labels)):
            w[labels[i]] = N_total / counts[i]
        # print('w:', w)
        w = w.cuda()

        # CELoss = nn.CrossEntropyLoss(weight=w)
        # loss_ce = CELoss(pred, target.cuda().long()).mean()

        loss_ce = F.cross_entropy(pred, target.cuda().long(), weight=w).mean()
        # loss_ce = F.cross_entropy(pred, target.cuda().long()).mean()
        # print(loss_ce)

        return loss_ce

class DiceLoss(nn.Module):
    def __init__(self, num_organ):
        super().__init__()
        self.num_organ  = num_organ

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)

        organ_target = torch.zeros((target.size(0), self.num_organ + 1, target.shape[-1], target.shape[-1])).cuda()
        # print('target_chiasm:', len(torch.where(target == 1)[0]))
        # print('target_pituitary:', len(torch.where(target == 2)[0]))

        for organ_index in range(self.num_organ + 1):
            temp_target = torch.zeros(target.size()).cuda()
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :] = temp_target
            # organ_target: (B, 14, 48, 256, 256)

        # print(len(idx_chiasm[0]))
        #
        # print('organ_target.chiasm:', organ_target[:, 1, :, :].sum(dim=[0, 1, 2]))
        # print('organ_target.pituitary:', organ_target[:, 2, :, :].sum(dim=[0, 1, 2]))
        #
        # print('pred.chiasm:', pred[:, 1, :, :].sum(dim=[0, 1, 2]))
        # print('pred.pituitary:', pred[:, 2, :, :].sum(dim=[0, 1, 2]))

        # 计算第一阶段的loss
        dice_stage1 = 0.0
        labels = torch.unique(target)
        labels = [int(l) for l in labels]
        dice_organ = np.zeros(self.num_organ + 1)

        # print(labels)
        for organ_index in range(self.num_organ + 1):
            if organ_index in labels:
                dice = 2 * (pred[:, organ_index, :, :] * organ_target[:, organ_index, :, :]).sum(dim=1).sum(dim=1) / \
                       (pred[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) +
                        organ_target[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)
                # print(organ_index, dice)
                dice_stage1 += dice

        # print(dice_stage1)

        dice_stage1 /= len(labels)
        # dice_stage1 /= (self.num_organ)
        # print((1 - dice_stage1).mean())
        # print('-------------------------')

        dice_loss = (1 - dice_stage1).mean()
        return dice_loss

class HybridLoss(nn.Module):
    def __init__(self, num_organ, size, resolution):
        super().__init__()
        self.num_organ  = num_organ
        self.size       = size
        self.resolution = resolution

    def forward(self, pred, target, size):
        organ_target = torch.zeros((target.size(0), self.num_organ + 1, size, size)).cuda()
        # print('target_chiasm:', len(torch.where(target == 1)[0]))
        # print('target_pituitary:', len(torch.where(target == 2)[0]))

        for organ_index in range(self.num_organ + 1):
            temp_target = torch.zeros(target.size()).cuda()
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :] = temp_target
            # organ_target: (B, 14, 48, 256, 256)

        # print(len(idx_chiasm[0]))
        #
        # print('organ_target.chiasm:', organ_target[:, 1, :, :].sum(dim=[0, 1, 2]))
        # print('organ_target.pituitary:', organ_target[:, 2, :, :].sum(dim=[0, 1, 2]))
        #
        # print('pred.chiasm:', pred[:, 1, :, :].sum(dim=[0, 1, 2]))
        # print('pred.pituitary:', pred[:, 2, :, :].sum(dim=[0, 1, 2]))

        # 计算第一阶段的loss
        dice_stage1 = 0.0
        labels = torch.unique(target)
        labels = [int(l) for l in labels]

        for organ_index in range(self.num_organ + 1):
            if organ_index in labels:
                dice = 2 * (pred[:, organ_index, :, :] * organ_target[:, organ_index, :, :]).sum(dim=1).sum(dim=1) / (pred[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) +
                                                         organ_target[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)
                # print(organ_index, dice)
                dice_stage1 += dice

        # print(dice_stage1)

        dice_stage1 /= len(labels)
        # dice_stage1 /= (self.num_organ)
        # print((1 - dice_stage1).mean())
        # print('-------------------------')
        return (1 - dice_stage1).mean()