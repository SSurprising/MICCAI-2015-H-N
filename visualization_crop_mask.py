import os
import numpy as np
import cv2 as cv
import SimpleITK as sitk
import skimage
from skimage import io, color, feature, measure
import random
import copy

join = os.path.join

# data_root = '/home/zjm/Data/HaN_OAR_raw/four_organ_fast_validate/subset5/'
# save_root = '/home/zjm/Data/HaN_OAR_raw/four_organ_crop_mask/'
# organ_map = ['bg', 'chiasm', 'pituitary', 'R', 'L']
#
# for patient in os.listdir(data_root):
#     print(save_root, patient)
#     save_path = os.path.join(save_root, patient)
#
#     label_path = os.path.join(data_root, patient, 'label.nii')
#
#     label = sitk.ReadImage(label_path)
#     label_array = sitk.GetArrayFromImage(label)
#
#     for slice_index in range(label_array.shape[0]):
#         slice = label_array[slice_index]
#         save_path_img = os.path.join(save_path, str(slice_index))
#         if not os.path.exists(save_path_img):
#             os.makedirs(save_path_img)
#
#         for organ_index in range(5):
#             mask = np.zeros(slice.shape)
#
#             idx = np.where(slice == organ_index)
#
#             mask[idx] = 255
#
#             cv.imwrite(os.path.join(save_path_img, organ_map[organ_index] + '.png'), mask)

data_root = '/home/zjm/Project/segmentation/BLSC_2D/outputs/val_subset5_SC_UNet_all_[128, 128]_celoss_expand2NumofOrgan_4/Exp2/pred_label/42.nii'
save_root = '/home/zjm/Data/HaN_OAR_raw/four_organ_crop_mask_42/'
organ_map = ['bg', 'chiasm', 'pituitary', 'R', 'L']

if True:

    label = sitk.ReadImage(data_root)
    label_array = sitk.GetArrayFromImage(label)

    for slice_index in range(label_array.shape[0]):
        slice = label_array[slice_index]
        save_path_img = os.path.join(save_root, str(slice_index))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)

        for organ_index in range(5):
            mask = np.zeros(slice.shape)

            idx = np.where(slice == organ_index)

            mask[idx] = 255
            center = [223, 256]
            mask = mask[223-64:223+64, 256-64:256+64]


            cv.imwrite(os.path.join(save_path_img, organ_map[organ_index] + '.png'), mask)
