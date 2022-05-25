import os
import numpy as np
import torch
import SimpleITK as sitk
from collections import Counter
import shutil

data_root = '/home/zjm/Data/MICCAI_2015/'

test_dir = os.path.join(data_root, 'test')
train_dir = os.path.join(data_root, 'train')

organ_num = []
all_list = []

organ_map = [
    'BrainStem',
    'Chiasm',
    'OpticNerve_L',
    'OpticNerve_R',
    'Parotid_L',
    'Parotid_R',
]
print(organ_map)

# check labels in dataset
# for root, dirs, files in os.walk(train_dir):
#     if 'structures' in root:
#         # print(len(files))
#         organ_num.append(len(files))
#
#         if len(files) == 6:
#             print(root)
#             print(files)
#
# for root, dirs, files in os.walk(test_dir):
#     if 'structures' in root:
#         # print(len(files))
#         organ_num.append(len(files))
#
#         if len(files) == 6:
#             print(root)
#             print(files)

# generate new labels with nii format
# for root, dirs, files in os.walk(data_root):
#     if 'structures' in root and ('train' in root or 'test' in root):
#         print(root)
#         save_path = root.split('structures')[0]
#
#         img = sitk.ReadImage(os.path.join(save_path, 'img.nrrd'))
#         img_array = sitk.GetArrayFromImage(img)
#
#         new_label_array = np.zeros_like(img_array)
#
#         for organ_index, organ in enumerate(organ_map, 1):
#             label = sitk.ReadImage(os.path.join(root, organ + '.nrrd'))
#             label_array = sitk.GetArrayFromImage(label)
#
#             new_label_array[label_array == 1] = organ_index
#
#         new_label = sitk.GetImageFromArray(new_label_array)
#
#         print(np.unique(new_label))
#         sitk.WriteImage(new_label, os.path.join(save_path, 'label.nii'))



new_save_path = os.path.join(data_root, '6organs')
if not os.path.exists(new_save_path):
    os.makedirs(new_save_path)

# make new dir to save data
# for root, dirs, files in os.walk(data_root):
#     if 'structures' in dirs and ('test' in root or 'train' in root):
#         p_name = root.split('/')[-1]
#         print(p_name)
#
#         if 'test' in root:
#             p_save_path = os.path.join(new_save_path, 'test', p_name)
#             if not os.path.exists(p_save_path):
#                 os.makedirs(p_save_path)
#
#             shutil.copy(os.path.join(root, 'img.nrrd'), p_save_path)
#             shutil.copy(os.path.join(root, 'label.nii'), p_save_path)
#         elif 'train' in root:
#             p_save_path = os.path.join(new_save_path, 'train', p_name)
#             if not os.path.exists(p_save_path):
#                 os.makedirs(p_save_path)
#
#             shutil.copy(os.path.join(root, 'img.nrrd'), p_save_path)
#             shutil.copy(os.path.join(root, 'label.nii'), p_save_path)

# change nrrd to nii
for root, dirs, files in os.walk(new_save_path):
    if 'img.nrrd' in files:
        img = sitk.ReadImage(os.path.join(root, 'img.nrrd'))
        img_array = sitk.GetArrayFromImage(img)
        print(img.GetSpacing())

        new_img = sitk.GetImageFromArray(img_array)
        new_img.SetSpacing(img.GetSpacing())
        sitk.WriteImage(new_img, os.path.join(root, 'img.nii'))


