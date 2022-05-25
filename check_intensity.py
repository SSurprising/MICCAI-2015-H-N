import os
import numpy as np
import torch
import SimpleITK as sitk
from collections import Counter
import matplotlib.pyplot as plt

data_dir = '/home/zjm/Data/MICCAI_2015/3organs/'

# 6 organs                  3 organs
# X: 145, 319               X:145 288
# Y: 161, 367               Y:217 313
# Center: [232, 264]        [216, 265]

chiasm_list  = []
nerve_L_list = []
nerve_R_list = []

dataset_list = ['train', 'test']
for subset in dataset_list:
    print('\nDataset:', subset)
    for patient in os.listdir(data_dir + subset):
        print(patient)
        image = sitk.ReadImage(data_dir + subset + '/' + patient + '/img.nii')
        label = sitk.ReadImage(data_dir + subset + '/' + patient + '/label.nii')

        img_arr = sitk.GetArrayFromImage(image)
        label_arr = sitk.GetArrayFromImage(label)

        labels = np.unique(label_arr)

        for organ_idx in range(1, len(labels)):
            idx = np.where(label_arr == organ_idx)

            for i in range(len(idx[0])):
                value = img_arr[idx[0][i], idx[1][i], idx[2][i]]

                if organ_idx == 1:
                    chiasm_list.append(value)
                elif organ_idx == 2:
                    nerve_L_list.append(value)
                elif organ_idx == 3:
                    nerve_R_list.append(value)

chiasm_list.sort()
nerve_L_list.sort()
nerve_R_list.sort()

chiasm_list = np.array(chiasm_list)
idx_chiasm_start = np.where(chiasm_list < -100)
idx_chiasm_end = np.where(chiasm_list > 100)

print(len(idx_chiasm_start[0]) / len(chiasm_list), len(idx_chiasm_end[0]) / len(chiasm_list))

nerve_L_list = np.array(nerve_L_list)
idx_nerve_L_start = np.where(nerve_L_list < -100)
idx_nerve_L_end = np.where(nerve_L_list > 100)

print(len(idx_nerve_L_start[0]) / len(nerve_L_list), len(idx_nerve_L_end[0]) / len(nerve_L_list))

nerve_R_list = np.array(nerve_R_list)
idx_nerve_R_start = np.where(nerve_R_list < -100)
idx_nerve_R_end = np.where(nerve_R_list > 100)

print(len(idx_nerve_R_start[0]) / len(nerve_R_list), len(idx_nerve_R_end[0]) / len(nerve_R_list))