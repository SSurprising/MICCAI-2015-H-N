# coding=UTF-8

import os
# from time import time
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
import SimpleITK as sitk
import numpy as np
from sklearn.model_selection import KFold

from loss import *
from dataset.dataload import Dataset_location
from utils.tools import *
from utils.parameter_parse import *

import logging

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

cudnn.benchmark = True

# SEED = 1
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser("The Parameters of the model.")

# Basic parameters
parser.add_argument('--model_name', type=str, default='BLSC', help='training model name')
parser.add_argument('--data_path', type=str, default='/home/zjm/Data/HaN_OAR_256/', help='the root path of data')
parser.add_argument('--val_subset', type=str, default='subset5', help='the directory name of the validation set in data_path, the rest of directory will be used as training set')
parser.add_argument('--repeat_times', type=str, default='1', help='the number of repeated experiments')
parser.add_argument('--load_all_image', type=bool, default=True)
parser.add_argument('--anno', type=str, default='')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--pin_memory', type=bool, default=False)

# Training parameters
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=1e-4, help='the original learning rate of training')
parser.add_argument('--slice_size', type=int, default=-1)
parser.add_argument('--resolution', type=int, nargs='+')
parser.add_argument('--crop_size', type=int, nargs='+')
parser.add_argument('--slice_expand', type=int, default=10, help='providing a random range when selecting slices')
parser.add_argument('--num_organ', type=int, default=22)
parser.add_argument('--HU_upper_threshold', type=int, default=-100)
parser.add_argument('--HU_lower_threshold', type=int, default=300)
parser.add_argument('--show_test_loss', action='store_true')
parser.add_argument('--loss', type=str, default='dice')
parser.add_argument('--auxiliary_loss', action='store_true')

# Resuming training parameters
parser.add_argument('--train', action='store_true')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the pretrained model')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
parser.add_argument('--MultiStepLR', type=int, nargs='+')
parser.add_argument('--lw', type=float, default=0.5, help='loss weight')

# Validating or testing parameters
parser.add_argument('--eval', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--test_file_dir', type=str, default='/home/zjm/Data/HaN_OAR_raw/chiasm_pituitary/subset5/')
parser.add_argument('--output_dir', type=str, default='./outputs/')
parser.add_argument('--test_model_path', type=str)
parser.add_argument('--test_latest', action='store_true')
parser.add_argument('--test_best_val', action='store_true')

args = parser.parse_args()
UID_name = '{}_{}_loss_expand{}'.format(args.model_name, args.loss, args.slice_expand, args.anno)

print(UID_name)
print(args)

organ_list = ['optic nerve R', 'optic nerve L']


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net = get_network(args.model_name, args.num_organ, args.train)
    save_path = make_path(args.output_dir, UID_name, 'Exp' + str(args.repeat_times))
    log_name = get_log_name(save_path, args.train)

    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG, filename=log_name, filemode='w')
    logging.info('%s', args)

    net = torch.nn.DataParallel(net).cuda()

    if args.test:
        logging.info('-----------------test--------------')
        fold_path = join(args.test_model_path, save_path)
        if args.test_latest:
            model_path = join(fold_path, 'latest.pth')
        elif args.test_best_val:
            model_path = join(fold_path, 'best_val.pth')
        else:
            model_path = join(fold_path, 'best.pth')

        model_pth = torch.load(model_path)
        net.load_state_dict(model_pth['state_dict'])

        train_loss_list = model_pth['train_loss_list']
        show_loss(train_loss_list, join(fold_path, 'training loss'))

        try:
            val_dice_list = model_pth['val_dice_list']
            show_loss(val_dice_list, join(fold_path, 'validate dice'))
        except:
            pass

        outputs_save_path = join(args.output_dir, save_path)
        test(net, outputs_save_path)
        print('-----------------------------------------------------')

    elif args.eval:
        logging.info('-----------------evaluate--------------')
        test(net, save_path=save_path)
    elif args.resume_training:
        # Note: loading the state dictionary should be after allocating gpu
        net.load_state_dict(torch.load(args.resume_model))
    elif args.train:
        ct_name_list, gt_name_list = load_name_list(args.data_path)

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Loading data')
        train_ct_list, train_gt_list = load_image(ct_name_list, gt_name_list, args.slice_expand)

        train_gt_list[train_gt_list == 1] = 0
        train_gt_list[train_gt_list == 2] = 1
        train_gt_list[train_gt_list == 3] = 2

        train_ds = Dataset_location(cts=train_ct_list, gts=train_gt_list)
        train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, pin_memory=args.pin_memory)

        logging.info('Loading end')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Loading end')

        loss_func = get_loss(args.loss, args.num_organ)

        # opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
        opt = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.99, nesterov=True)

        # lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=20, min_lr=1e-8,
        #                                                       verbose=True)
        min_loss = 10
        best_val_dice = 0
        train_loss_list = []
        val_dice_list = []
        best_epoch = 0

        # model_save_path = make_path(save_path, 'Exp' + str(args.repeat_times))

        logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Begin training')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Begin training')
        for epoch in range(1, args.epochs + 1):
            train_loss = train(net, train_dl, loss_func, opt)
            # lr_decay.step(train_loss)
            opt.param_groups[0]['lr'] = args.learning_rate * (1-epoch/args.epochs)**0.9

            train_loss_list.append(train_loss)

            logging.info('epoch: [%d], lr [%.8f], train_loss: [%.8f], min_loss: [%.8f], best_epoch: [%d], best_val_dice: [%.2f]',
                         epoch, opt.state_dict()['param_groups'][0]['lr'], train_loss, min_loss, best_epoch, best_val_dice)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                  'epoch:{}, lr:{}, train_loss:{}, min_loss:{}, best_epoch:{}, best_val_dice:{}'.format(
                      epoch, opt.state_dict()['param_groups'][0]['lr'], train_loss, min_loss, best_epoch, best_val_dice))

            if args.show_test_loss:
                with torch.no_grad():
                    val_dice = test(net)
                val_dice_list.append(val_dice)
                if best_val_dice < val_dice and epoch > 10:
                    best_val_dice = val_dice
                    best_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'min_val_loss': min_loss,
                        'lr': opt.state_dict()['param_groups'][0]['lr'],
                        'train_loss_list': train_loss_list,
                        'val_dice_list': val_dice_list,
                        'state_dict': net.state_dict(),
                    }, join(save_path, 'best_val.pth'))
            else:
                if train_loss < min_loss and epoch > 10:
                    min_loss = train_loss
                    best_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'min_val_loss': min_loss,
                        'lr': opt.state_dict()['param_groups'][0]['lr'],
                        'train_loss_list': train_loss_list,
                        'val_dice_list': val_dice_list,
                        'state_dict': net.state_dict(),
                    }, join(save_path, 'best.pth'))

            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'min_val_loss': min_loss,
                    'lr': opt.state_dict()['param_groups'][0]['lr'],
                    'train_loss_list': train_loss_list,
                    'val_dice_list': val_dice_list,
                    'state_dict': net.state_dict(),
                }, join(save_path, 'latest.pth'))


def train(net, train_dl, loss_func, opt):
    net.train()

    mean_loss = []
    for step, (ct, seg) in enumerate(train_dl):
        loss = 0

        outputs = net(ct.cuda())

        if args.auxiliary_loss:
            loss += auxiliary_loss(loss_func, outputs, seg)
        else:
            loss += loss_func(outputs[-1], seg.cuda())

        mean_loss.append(loss.item())
        loss.backward()
        opt.step(loss.item)

        opt.zero_grad()

    mean_loss = sum(mean_loss) / len(mean_loss)
    return mean_loss


def test(net, save_path=None):
    net.eval()
    dice_list = []
    dice_list_pp = []

    for root, dirs, files in os.walk(args.data_path):
        if len(files) and 'img.nii' in files:
            print(root)
            seg_list = []

            img = sitk.ReadImage(os.path.join(root, 'img.nii'))
            gt = sitk.ReadImage(os.path.join(root, 'label.nii'))

            img_arr = sitk.GetArrayFromImage(img)
            gt_arr = sitk.GetArrayFromImage(gt)

            gt_arr[gt_arr == 1] = 0
            gt_arr[gt_arr == 2] = 1
            gt_arr[gt_arr == 3] = 2
            # print('gt_arr.shape', gt_arr.shape)

            img_arr[img_arr > args.HU_upper_threshold] = args.HU_upper_threshold
            img_arr[img_arr < args.HU_lower_threshold] = args.HU_lower_threshold

            if args.HU_upper_threshold == abs(args.HU_lower_threshold):
                img_arr = img_arr / args.HU_upper_threshold
            else:
                img_arr = (img_arr - args.HU_lower_threshold) / (args.HU_upper_threshold - args.HU_lower_threshold)

            # print('label img_arr', np.unique(img_arr))
            with torch.no_grad():
                img_arr = np.array(img_arr).squeeze()
                for i in range(img_arr.shape[0]):
                    slice = img_arr[i, :, :]
                    slice = torch.FloatTensor(slice).cuda().unsqueeze(dim=0).unsqueeze(dim=0)

                    seg_result = net(slice).squeeze(dim=0).cpu().detach().numpy()

                    seg_result = np.argmax(seg_result, axis=0)
                    seg_list.append(seg_result)

                pred_seg = np.array(seg_list).squeeze()

            pred_img = sitk.GetImageFromArray(pred_seg)

            patient_name = root.split('/')[-1]
            pred_save_path = make_path(save_path, 'pred_label')
            sitk.WriteImage(pred_img, join(pred_save_path, patient_name + '.nii'))
            # print('save path:', join(pred_save_path, patient_name + '.nii'))

            # calculate the dice score for each patient
            dice_score_patient = dice_score(pred_seg, gt_arr, labels=[i for i in range(1, args.num_organ + 1)])
            dice_score_patient = [round(d * 100, 2) for d in dice_score_patient]
            print('dice w/o pp:', dice_score_patient)
            dice_list.append(dice_score_patient)

            pred_seg_pp = remove_small_region_3d(pred_seg)
            dice_score_patient_pp = dice_score(pred_seg_pp, gt_arr,
                                               labels=[i for i in range(1, args.num_organ + 1)])
            dice_score_patient_pp = [round(d * 100, 2) for d in dice_score_patient_pp]
            print('dice w/  pp:', dice_score_patient_pp)
            dice_list_pp.append(dice_score_patient_pp)

    # axis=0 means to calculate the average value by column
    mean_dice = np.array(dice_list).mean(axis=0)
    mean_dice_pp = np.array(dice_list_pp).mean(axis=0)

    mean_dice_average = np.array(mean_dice).mean()
    mean_dice_pp_average = np.array(mean_dice_pp).mean()
    print('mean dice w/o pp:', mean_dice, 'average:', mean_dice_average)
    print('mean dice w/  pp:', mean_dice_pp, 'average:', mean_dice_pp_average)

    return mean_dice


def auxiliary_loss(loss_func, outputs, seg, loss_weight=[1/7, 2/7, 4/7, 1]):
    loss = 0
    seg_4 = seg
    seg_3 = F.interpolate(seg.unsqueeze(dim=0), [seg.shape[1] // 2, seg.shape[2] // 2]).squeeze(dim=0)
    seg_2 = F.interpolate(seg.unsqueeze(dim=0), [seg.shape[1] // 4, seg.shape[2] // 4]).squeeze(dim=0)
    seg_1 = F.interpolate(seg.unsqueeze(dim=0), [seg.shape[1] // 8, seg.shape[2] // 8]).squeeze(dim=0)
    segs = [seg_1, seg_2, seg_3, seg_4]

    for i in range(len(loss_weight)):
        loss += loss_weight[i] * loss_func(outputs[i], segs[i].cuda())

    return loss


if __name__ == '__main__':
    main()
