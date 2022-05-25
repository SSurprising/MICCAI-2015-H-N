import os
import numpy as np
import torch
import SimpleITK as sitk
from collections import Counter
import scipy.ndimage as ndimage

data_dir = '/home/zjm/Data/MICCAI_2015/3organs/'

target_spacing = [1.11, 1.11, 3]


def resample(img, target_spacing, order=3, cval=0, target_size=512):
    img_array = sitk.GetArrayFromImage(img)

    if img.GetSpacing()[-1] / img.GetSpacing()[0] > 3:
        new_img_array = ndimage.zoom(img_array, (1, img.GetSpacing()[0] / target_spacing[0], img.GetSpacing()[1] /
                                                 target_spacing[1]), order=order)
        new_img_array = ndimage.zoom(img_array, (img.GetSpacing()[2] / target_spacing[2], 1, 1), order=order)
    else:
        new_img_array = ndimage.zoom(img_array, (img.GetSpacing()[2] / target_spacing[2], img.GetSpacing()[0] /
                                                 target_spacing[0], img.GetSpacing()[1] / target_spacing[1]), order=order)

    print('origin shape', img_array.shape, 'resample shape', new_img_array.shape)

    if new_img_array.shape[-1] < target_size:
        final_image_array = np.full((new_img_array.shape[0], target_size, target_size), cval)
        start_coords = (target_size - new_img_array.shape[1]) // 2
        end_coords = start_coords + new_img_array.shape[1]

        final_image_array[:, start_coords:end_coords, start_coords:end_coords] = new_img_array
    elif new_img_array.shape[-1] > target_size:
        final_image_array = np.full((new_img_array.shape[0], target_size, target_size), cval)

        start_coords = (new_img_array.shape[1] - target_size) // 2
        end_coords = start_coords + target_size

        final_image_array = new_img_array[:, start_coords:end_coords, start_coords:end_coords]
    else:
        return new_img_array

    return final_image_array



for root, dirs, files in os.walk(data_dir):
    if 'img.nii' in files:
        print(root)
        new_path = root.replace('3organs/', '3organs/preprocess/')

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        img = sitk.ReadImage(os.path.join(root, 'img.nii'))
        label = sitk.ReadImage(os.path.join(root, 'label.nii'))

        new_img_array = resample(img, target_spacing, order=3, cval=-1024)
        new_label_array = resample(label, target_spacing, order=0, cval=0)

        final_img = sitk.GetImageFromArray(new_img_array)
        final_label = sitk.GetImageFromArray(new_label_array)

        final_img.SetSpacing(target_spacing)
        final_label.SetSpacing(target_spacing)

        sitk.WriteImage(final_img, os.path.join(new_path,  'img.nii'))
        sitk.WriteImage(final_label, os.path.join(new_path, 'label.nii'))




