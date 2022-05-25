import os
import numpy as np
import torch
import SimpleITK as sitk
from collections import Counter

# target spacing = 1.11 * 1.11 * 3

data_dir = '/home/zjm/Data/MICCAI_2015/3organs/'

median_shape_x = []
median_shape_z = []

for root, dirs, files in os.walk(data_dir):
    if 'img.nii' in files:
    # if 'img.nii' in files and 'train' in root:
        print(root)
        label = sitk.ReadImage(os.path.join(root, 'label.nii'))
        label_array = sitk.GetArrayFromImage(label)

        img = sitk.ReadImage(os.path.join(root, 'img.nii'))
        spacing = img.GetSpacing()

        # labels = np.unique(label_array).astype(np.int)

        idx_x, idx_y, idx_z = np.where(label_array != 0)

        idx_x = max(idx_x) - min(idx_x) + 1
        idx_y = max(idx_y) - min(idx_y) + 1
        idx_z = max(idx_z) - min(idx_z) + 1

        print(idx_x, idx_y, idx_z, img.GetSpacing())
        median_shape_x.append(np.round(img.GetSpacing()[0], 2))
        median_shape_z.append(np.round(img.GetSpacing()[-1], 2))

mid = len(median_shape_x) // 2
# print(np.array(median_shape_x).mean())
print(Counter(median_shape_x))
median_shape_x.sort()
print(median_shape_x[mid-1], median_shape_x[mid], median_shape_x[mid+1])

# print(np.array(median_shape_z).mean())
print(Counter(median_shape_z))
median_shape_z.sort()
print(median_shape_z[mid])
print(median_shape_z[mid-1], median_shape_z[mid], median_shape_z[mid+1])


