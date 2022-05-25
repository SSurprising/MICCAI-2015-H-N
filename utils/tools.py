import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os
import SimpleITK as sitk
import skimage
from skimage import measure
from skimage.measure import regionprops
from collections import OrderedDict

join = os.path.join


def make_path(root, *args):
    new_path = root

    for i in range(len(args)):
        new_path = join(new_path, args[i])

    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path


def dice_score(pred, seg, labels):
    dice_list = []
    for organ_index in labels:
        pred_organ   = np.zeros(pred.shape)
        target_organ = np.zeros(seg.shape)

        pred_organ[pred == organ_index] = 1
        target_organ[seg == organ_index] = 1

        dice_organ = (2 * pred_organ * target_organ).sum() / (pred_organ.sum() + target_organ.sum())
        dice_list.append(dice_organ)

    return dice_list


def image_plot(x, y, save_path, color, label, title='Loss with epochs', x_label='epochs', y_label='loss', mode=0):
    for i in range(len(y)):
        plt.plot(x[i], y[i], color=color[i], label=label[i], linewidth=1, marker='o', markerfacecolor='red', markersize=1)


    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # omit the redundant legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualization_slice(img, save_path, label=None):
    for slice_index in range(0, img.shape[0]):
        img_slice = img[slice_index] * 255
        pass


def crop_image(image, size=256, center=[236, 256], keep_center_block=True, random_flag=False, one_time_crop=True):
    new_img_list = []
    if one_time_crop:
        if random_flag:
            x_min = random.randint(center[0] - size, center[0])
            y_min = random.randint(center[1] - size, center[1])
        else:
            x_min = int(center[0] - size / 2)
            y_min = int(center[1] - size / 2)

        if len(image.shape) == 3:
            new_img_list.append(image[:, x_min:x_min+size, y_min:y_min+size])
        elif len(image.shape) == 2:
            new_img_list.append(image[x_min:x_min+size, y_min:y_min+size])

        return new_img_list, [x_min, y_min]
    else:
        pass


def load_all_image(data_path, slice_expand=-1, HU_upper=300, HU_lower=-100, val_subset='subset5'):
    cts = np.zeros((1, 512, 512))
    gts = np.zeros((1, 512, 512))

    print("**********Loading all images and labels**********")
    for root, dirs, files in os.walk(data_path):
        if (val_subset not in root) and 'image.nii' in files:
            ct = sitk.ReadImage(os.path.join(root, 'image.nii'))
            ct_arr = sitk.GetArrayFromImage(ct)
            ct_arr[ct_arr > HU_upper] = HU_upper
            ct_arr[ct_arr < HU_lower] = HU_lower
            ct_arr = (ct_arr - HU_lower) / (HU_upper - HU_lower)

            gt = sitk.ReadImage(os.path.join(root, 'label.nii'))
            gt_arr = sitk.GetArrayFromImage(gt)

            idx = np.where(gt_arr != 0)

            if slice_expand == -1:
                cts.append(ct_arr)
                gts.append(gt_arr)
            else:
                idx_min = np.min(idx[0]) - slice_expand
                idx_max = np.max(idx[0]) + slice_expand

                idx_max = min(gt_arr.shape[0]-1, idx_max)

                cts = np.concatenate([cts, ct_arr[idx_min:idx_max + 1, :, :]])
                gts = np.concatenate([gts, gt_arr[idx_min:idx_max + 1, :, :]])

    print("**********Loading end**********")
    return cts[1:, :, :], gts[1:, :, :]


def load_name_list(data_path):
    ct_name_list = []
    gt_name_list = []
    print(data_path)
    for root, dirs, files in os.walk(data_path):
        if 'img.nii' in files and 'train' in root:
            ct_name_list.append(join(root, 'img.nii'))
            gt_name_list.append(join(root, 'label.nii'))

    return ct_name_list, gt_name_list


def load_image(ct_name_list, gt_name_list, slice_expand=4, index_list=None, hu_upper=100, hu_lower=-100):
    if index_list is None:
        index_list = range(len(ct_name_list))

    ct_list = np.zeros((1, 512, 512))
    gt_list = np.zeros((1, 512, 512))

    for idx in index_list:
        ct = sitk.ReadImage(ct_name_list[idx])
        gt = sitk.ReadImage(gt_name_list[idx])

        ct_array = sitk.GetArrayFromImage(ct)
        gt_array = sitk.GetArrayFromImage(gt)

        ct_array[ct_array > hu_upper] = hu_upper
        ct_array[ct_array < hu_lower] = hu_lower

        if hu_upper == abs(hu_lower):
            ct_array = ct_array / hu_upper
        else:
            ct_array = (ct_array - hu_lower) / (hu_upper - hu_lower)

        # ct_list = np.concatenate((ct_list, ct_array), axis=0)
        # gt_list = np.concatenate((gt_list, gt_array), axis=0)

        if slice_expand > -1:
            idx = np.where(gt_array != 0)
            idx_min = np.min(idx[0]) - slice_expand
            idx_max = np.max(idx[0]) + slice_expand
            idx_max = min(gt_array.shape[0] - 1, idx_max)

            ct_list = np.concatenate([ct_list, ct_array[idx_min:idx_max + 1, :, :]])
            gt_list = np.concatenate([gt_list, gt_array[idx_min:idx_max + 1, :, :]])
        else:
            ct_list = np.concatenate([ct_list, ct_array])
            gt_list = np.concatenate([gt_list, gt_array])

    return ct_list[1:, ::], gt_list[1:, ::]


def remove_small_region_2d(pred_label, threshold=[30, 10]):
    pred_label = np.int64(pred_label)
    labels = np.unique(pred_label)
    for label in labels:
        if label:
            idx   = np.where(pred_label == label)
            count = len(idx[0])
            if count < threshold[label-1]:
                pred_label[pred_label == label] = 0

    return pred_label


def remove_small_region_3d(pred):
    pred = np.int64(pred)
    labels = np.unique(pred)
    # print('labels:', labels)
    # print('pred_sum', pred.sum()/labels[-1])
    new_pred = np.zeros(pred.shape)

    for label in labels:
        if label:
            pred_temp = np.zeros(pred.shape)
            pred_temp[pred == label] = 1
            pred_temp = keep_largest_region_3d(pred_temp, label)

            new_pred += pred_temp

    return new_pred


# only for image with one label
def keep_largest_region_3d(pred, label_value):
    label, num = measure.label(pred, return_num=True)
    pred_temp = np.zeros(pred.shape)

    if num < 1:
        return pred

    areas = [region.area for region in regionprops(label)]
    areas.sort()
    # print(areas)

    for r in regionprops(label):
        if r.area == areas[-1]:
            pred_temp[r.coords[:, 0], r.coords[:, 1], r.coords[:, 2]] = label_value

    return pred_temp


def gen_train_data(cts, gts):
    pass


def visualization(ct, seg, save_path):
    # BGR
    color_map = [
        [0, 0, 255],    # red   chiasm
        [0, 255, 0],    # green pituitary
        [255, 0, 0],    # blue  nerve
        [0, 165, 255],  # orange nerve
    ]

    gray = np.zeros((ct.shape[-2], ct.shape[-1], 3))
    gray[::, 0] = ct
    gray[::, 1] = ct
    gray[::, 2] = ct


def show_loss(loss_list, save_path, color='red', label='training loss', x_label='epochs', y_label='loss', title=None):
    x = range(1, len(loss_list)+1)
    plt.plot(x, loss_list, color=color, label=label, linewidth=1, marker='o', markerfacecolor='red', markersize=1)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # # omit the redundant legend
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()