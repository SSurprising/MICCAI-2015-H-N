#coding=UTF-8
 
import os
#from time import time
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
from dataset.dataload import Dataset
from utils.tools import *
from utils.parameter_parse import *

import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cudnn.benchmark = True

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
parser.add_argument('--crop_image_one_time', type=bool, default=True)
parser.add_argument('--show_test_loss', action='store_true')
parser.add_argument('--loss', type=str, default='dice')

# Resuming training parameters
parser.add_argument('--train', action='store_true')
parser.add_argument('--flooding', type=float, default=None)
parser.add_argument('--resume_training', action='store_true', help='Resume training from the pretrained model')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
parser.add_argument('--MultiStepLR', type=int, nargs='+')
parser.add_argument('--ensemble', type=int, default=0)
parser.add_argument('--fold', type=int, default=1)

# Validating or testing parameters
parser.add_argument('--eval', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--test_file_dir', type=str, default='/home/zjm/Data/HaN_OAR_raw/chiasm_pituitary/subset5/')
parser.add_argument('--output_dir', type=str, default='./outputs/')
parser.add_argument('--test_model_path', type=str)
parser.add_argument('--test_latest', action='store_true')

args = parser.parse_args()
UID_name = 'val_{}_{}_{}_{}loss_expand{}'.format(args.val_subset, args.model_name, args.crop_size, args.loss,
                                                         args.slice_expand, args.anno)

# UID_name = 'val_' + args.val_subset + '_' + args.model_name + "_" + str(args.slice_size)
UID_name += 'NumofOrgan_' + str(args.num_organ)
print(UID_name)
print(args)

organ_list = ['chiasm', 'pituitary', 'optic nerve R', 'optic nerve L']


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net = get_network(args.model_name, args.num_organ, args.train)
    save_path = make_path(args.output_dir, UID_name, 'Exp' + str(args.repeat_times))
    log_name = get_log_name(save_path, args.train, args.ensemble)

    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG, filename=log_name, filemode='w')
    logging.info('%s', args)

    net = torch.nn.DataParallel(net).cuda()

    if args.test:
        logging.info('-----------------test--------------')
        if args.ensemble:
            for fold in range(1, args.ensemble+1):
                logging.info('-----------------fold %d--------------', fold)
                fold_path = join(args.test_model_path, str(fold))

                if args.test_latest:
                    model_path = join(fold_path, 'latest.pth')
                else:
                    model_path = join(fold_path, 'best.pth')

                model_pth = torch.load(model_path)
                net.load_state_dict(model_pth['state_dict'])

                train_loss_list = model_pth['train_loss_list']
                show_loss(train_loss_list, join(fold_path, 'training loss'))
                val_loss_list = model_pth['val_loss_list']
                show_loss(val_loss_list, join(fold_path, 'testing loss'))

                outputs_save_path = join(args.output_dir, str(fold))
                test(net, outputs_save_path)
                print('-----------------------------------------------------')
        else:
            fold_path = join(args.test_model_path, save_path)
            if args.test_latest:
                model_path = join(fold_path, 'latest.pth')
            else:
                model_path = join(fold_path, 'best.pth')

            model_pth = torch.load(model_path)
            net.load_state_dict(model_pth['state_dict'])

            train_loss_list = model_pth['train_loss_list']
            show_loss(train_loss_list, join(fold_path, 'training loss'))

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
        if args.ensemble:
            logging.info('Train models on %s sub-folds and ensemble them', str(args.ensemble))
            ct_name_list, gt_name_list = load_name_list(args.data_path, mute_name=args.val_subset)

            kf = KFold(n_splits=args.ensemble)
            kf.get_n_splits(ct_name_list)

            fold = 1
            for train_index, val_index in kf.split(ct_name_list):
                # if args.fold != fold:
                #     fold += 1
                #     continue

                logging.info('Loading data for fold [%d]', fold)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Loading data for fold [%d]', fold)
                train_ct_list, train_gt_list = load_image(ct_name_list, gt_name_list, train_index)
                val_ct_list, val_gt_list = load_image(ct_name_list, gt_name_list, val_index)

                train_ds = Dataset(cts=train_ct_list, gts=train_gt_list, crop_size=args.crop_size)
                train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, pin_memory=args.pin_memory)

                val_ds = Dataset(cts=val_ct_list, gts=val_gt_list, crop_size=args.crop_size)
                val_dl = DataLoader(val_ds, args.batch_size, True, num_workers=args.num_workers, pin_memory=args.pin_memory)

                logging.info('Loading end')
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Loading end')

                loss_func = get_loss(args.loss, args.num_organ)
                opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
                lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=30,
                                                                      min_lr=1e-8,
                                                                      verbose=True, threshold=1e-4,
                                                                      threshold_mode='abs')
                min_val_loss = 10
                train_loss_list = []
                val_loss_list   = []

                model_save_path = make_path(save_path, str(fold))

                logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Begin training')
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Begin training')
                for epoch in range(1, args.epochs + 1):
                    train_loss = train(net, train_dl, loss_func, opt)
                    val_loss   = val(net, val_dl, loss_func)
                    lr_decay.step(val_loss)

                    train_loss_list.append(train_loss)
                    val_loss_list.append(val_loss)

                    logging.info('fold: [%d], epoch: [%d], train_loss: [%.8f], val_loss: [%.8f]',
                                 fold, epoch, train_loss, val_loss)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                          'fold:{}, epoch:{}, train_loss:{}, val_loss:{}'.format(fold, epoch, train_loss, val_loss))

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        torch.save({
                            'fold': fold,
                            'epoch': epoch,
                            'min_val_loss': min_val_loss,
                            'lr': opt.state_dict()['param_groups'][0]['lr'],
                            'train_loss_list': train_loss_list,
                            'val_loss_list': val_loss_list,
                            'state_dict': net.state_dict(),
                        }, join(model_save_path, 'best.pth'))
                    else:
                        torch.save({
                            'fold': fold,
                            'epoch': epoch,
                            'min_val_loss': min_val_loss,
                            'lr': opt.state_dict()['param_groups'][0]['lr'],
                            'train_loss_list': train_loss_list,
                            'val_loss_list': val_loss_list,
                            'state_dict': net.state_dict(),
                        }, join(model_save_path, 'latest.pth'))

                fold += 1
        else:
            logging.info('Train one model without ensemble')
            ct_name_list, gt_name_list = load_name_list(args.data_path, mute_name=args.val_subset)

            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Loading data')
            train_ct_list, train_gt_list = load_image(ct_name_list, gt_name_list, args.slice_expand)

            train_ds = Dataset(cts=train_ct_list, gts=train_gt_list, crop_size=args.crop_size)
            train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, pin_memory=args.pin_memory)

            logging.info('Loading end')
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Loading end')

            loss_func = get_loss(args.loss, args.num_organ)
            opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

            lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=30,
                                                                  min_lr=1e-8, verbose=True, threshold=1e-4, threshold_mode='abs')
            # lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=30, min_lr=1e-8,
            #                                                       verbose=True)
            min_loss = 10
            train_loss_list = []

            # model_save_path = make_path(save_path, 'Exp' + str(args.repeat_times))

            logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Begin training')
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'Begin training')
            for epoch in range(1, args.epochs + 1):
                train_loss = train(net, train_dl, loss_func, opt)
                lr_decay.step(train_loss)

                train_loss_list.append(train_loss)

                logging.info('epoch: [%d], train_loss: [%.8f]', epoch, train_loss)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'epoch:{}, train_loss:{}'.format(epoch, train_loss))

                if train_loss < min_loss:
                    min_loss = train_loss
                    torch.save({
                        'epoch': epoch,
                        'min_val_loss': min_loss,
                        'lr': opt.state_dict()['param_groups'][0]['lr'],
                        'train_loss_list': train_loss_list,
                        'state_dict': net.state_dict(),
                    }, join(save_path, 'best.pth'))

                if epoch % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'min_val_loss': min_loss,
                        'lr': opt.state_dict()['param_groups'][0]['lr'],
                        'train_loss_list': train_loss_list,
                        'state_dict': net.state_dict(),
                    }, join(save_path, 'latest.pth'))


def train(net, train_dl, loss_func, opt):
    net.train()

    mean_loss = []
    for step, (ct, seg) in enumerate(train_dl):
        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)
        loss = loss_func(outputs, seg)

        mean_loss.append(loss.item())
        loss.backward()
        opt.step(loss.item)

        opt.zero_grad()

    mean_loss = sum(mean_loss) / len(mean_loss)
    return mean_loss

    #     # check the time and draw the loss image every 10 epochs
    #     if epoch % 10 == 0:
    #         print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    #         torch.save(net.state_dict(), os.path.join(model_save_path, 'latest_backup.pth'))
    #
    #         x1 = range(1, epoch + 1)
    #         # y_flood = np.full(epoch, args.flooding)
    #
    #         if args.show_test_loss:
    #             image_plot([x1, x2], y=[loss_list, test_loss1],
    #                        save_path=os.path.join(save_path, 'loss.png'),
    #                        color=['orange', 'red'], label=['training loss', 'test loss'])
    #         else:
    #             image_plot([x1], y=[loss_list], save_path=os.path.join(save_path, 'loss.png'),
    #                        color=['orange'], label=['training loss'])
    #
    #     print('epoch:{}, time:{:.3f} min'.format(epoch, (time.time() - start) / 60))
    #     print('************************')
    #     print()
    #
    # print('min_loss = ', min_loss)


def val(net, val_dl, loss_func):
    net.eval()

    with torch.no_grad():
        mean_loss = []
        for step, (ct, seg) in enumerate(val_dl):
            ct = ct.cuda()
            seg = seg.cuda()

            outputs = net(ct)
            loss = loss_func(outputs, seg)

        mean_loss.append(loss.item())

    mean_loss = sum(mean_loss) / len(mean_loss)

    return mean_loss


def test(net, save_path=None, size=128):
    net.eval()
    dice_list = []
    dice_list_pp = []

    for root, dirs, files in os.walk(args.data_path):
        if len(files) and args.val_subset in root:
            print(root)
            seg_list = []

            img = sitk.ReadImage(os.path.join(root, 'image.nii'))
            gt  = sitk.ReadImage(os.path.join(root, 'label.nii'))

            img_arr = sitk.GetArrayFromImage(img)
            gt_arr  = sitk.GetArrayFromImage(gt)
            # print('gt_arr.shape', gt_arr.shape)

            img_arr[img_arr > args.HU_upper_threshold] = args.HU_upper_threshold
            img_arr[img_arr < args.HU_lower_threshold] = args.HU_lower_threshold

            # print('label img_arr', np.unique(img_arr))
            with torch.no_grad():
                # new_img_arr, [x_min, y_min] = crop_image(img_arr, center=[223, 295])
                new_img_arr, [x_min, y_min] = crop_image(img_arr, center=[223, 256])
                new_img_arr = np.array(new_img_arr).squeeze()
                for i in range(new_img_arr.shape[0]):
                    slice = new_img_arr[i, :, :]
                    slice = torch.FloatTensor(slice).cuda().unsqueeze(dim=0).unsqueeze(dim=0)

                    seg_result = net(slice).squeeze(dim=0).cpu().detach().numpy()

                    seg_result = np.argmax(seg_result, axis=0)
                    seg_list.append(seg_result)

                pred_seg = np.array(seg_list).squeeze()

            pred_seg_full = np.zeros(gt_arr.shape)

            pred_seg_full[:, x_min:x_min+size, y_min:y_min+size] = pred_seg
            pred_img = sitk.GetImageFromArray(pred_seg_full)

            patient_name = root.split('/')[-1]
            pred_save_path = make_path(save_path, 'pred_label')
            sitk.WriteImage(pred_img, join(pred_save_path, patient_name + '.nii'))
            print('save path:', join(pred_save_path, patient_name + '.nii'))

            # calculate the dice score for each patient
            dice_score_patient = dice_score(pred_seg_full, gt_arr, labels=[i for i in range(1, args.num_organ + 1)])
            dice_score_patient = [round(d*100, 2) for d in dice_score_patient]
            print('dice w/o pp:', dice_score_patient)
            dice_list.append(dice_score_patient)

            pred_seg_full_pp = remove_small_region_3d(pred_seg_full)
            dice_score_patient_pp = dice_score(pred_seg_full_pp, gt_arr, labels=[i for i in range(1, args.num_organ + 1)])
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


if __name__ =='__main__':
    main()