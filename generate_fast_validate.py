import os
import numpy as np
import torch
import SimpleITK as sitk
from collections import Counter

data_dir = '/home/zjm/Data/HaN_OAR_raw/four_organ/'

new_data_dir = data_dir.replace('four_organ', 'four_organ_fast_validate')

for subset in os.listdir(data_dir):
    for patient in os.listdir(data_dir + subset):
        print(patient)
        save_path = os.path.join(new_data_dir, subset, patient)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image = sitk.ReadImage(data_dir + subset + '/' + patient + '/image.nii')
        label = sitk.ReadImage(data_dir + subset + '/' + patient + '/label.nii')

        img_arr = sitk.GetArrayFromImage(image)
        gt_arr = sitk.GetArrayFromImage(label)

        idx = np.where(gt_arr != 0)
        img_arr = img_arr[min(idx[0]):max(idx[0]) + 1, ::]
        gt_arr = gt_arr[min(idx[0]):max(idx[0]) + 1, ::]

        img_arr[img_arr > 300] = 300
        img_arr[img_arr < -100] = -100

        center = [223, 256]
        x_min = int(center[0] - 64)
        y_min = int(center[1] - 64)

        new_img_arr = img_arr[:, x_min:x_min+128, y_min:y_min+128]
        new_gt_arr  = gt_arr[:, x_min:x_min+128, y_min:y_min+128]
        print(new_img_arr.shape, new_gt_arr.shape)
        new_img = sitk.GetImageFromArray(new_img_arr)

        new_label = sitk.GetImageFromArray(new_gt_arr)

        sitk.WriteImage(new_img, os.path.join(save_path, 'image.nii'))
        sitk.WriteImage(new_label, os.path.join(save_path, 'label.nii'))







