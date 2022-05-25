import os
import numpy as np
import torch
import SimpleITK as sitk
from collections import Counter
import shutil

data_root = '/home/zjm/Data/MICCAI_2015/6organs/'

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


for root, dirs, files in os.walk(data_root):
    if 'img.nii' in files:
        print(root)
        new_path = root.replace('6organs', '3organs')
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        shutil.copy(os.path.join(root, 'img.nii'), new_path)

        img = sitk.ReadImage(os.path.join(root, 'img.nii'))

        label = sitk.ReadImage(os.path.join(root, 'label.nii'))
        label_array = sitk.GetArrayFromImage(label)

        label_array[label_array == 1] = 0
        label_array[label_array == 5] = 0
        label_array[label_array == 6] = 0

        label_array[label_array == 2] = 1
        label_array[label_array == 3] = 2
        label_array[label_array == 4] = 3

        new_label = sitk.GetImageFromArray(label_array)
        new_label.SetSpacing(img.GetSpacing())

        sitk.WriteImage(new_label, os.path.join(new_path, 'label.nii'))


