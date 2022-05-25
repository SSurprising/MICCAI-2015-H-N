import os
import numpy as np
import cv2 as cv
import SimpleITK as sitk
import skimage
from skimage import io, color, feature, measure
import random
import copy

join = os.path.join

print('====================================================')
print('visualizing the raw images')
print('====================================================')

# BGR order
color_map = [
    [0, 255, 0],    # Green            --chiasm
    [0, 0, 255],    # Red              --pituitary
    [0, 165, 255],  # Orange           --nerve L
    [255, 0, 0],    # Blue             --nerve R
]

HU_upper = 300
HU_lower = -100

ct_path_4organ = '/home/zjm/Data/HaN_OAR_raw/four_organ/'
save_path = '/home/zjm/Data/HaN_OAR_raw/visualization/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for root, dirs, files in os.walk(ct_path_4organ):
    if len(files) and files[0].endswith('.nii'):
        print(root)
        p_name = root.split('/')[-1]

        p_save_path = join(save_path, p_name)
        if not os.path.exists(p_save_path):
            os.makedirs(p_save_path)

        gray_save_path = join(p_save_path, 'gray')
        if not os.path.exists(gray_save_path):
            os.makedirs(gray_save_path)

        ct_save_path_crop = join(p_save_path, 'ct_crop')
        if not os.path.exists(ct_save_path_crop):
            os.makedirs(ct_save_path_crop)

        mask_save_path_crop = join(p_save_path, 'mask_crop')
        if not os.path.exists(mask_save_path_crop):
            os.makedirs(mask_save_path_crop)

        img_ct = sitk.ReadImage(join(root, 'image.nii'))
        img_gt = sitk.ReadImage(join(root, 'label.nii'))

        ct_array = sitk.GetArrayFromImage(img_ct)
        gt_array = sitk.GetArrayFromImage(img_gt)

        ct_array[ct_array > HU_upper] = HU_upper
        ct_array[ct_array < HU_lower] = HU_lower

        # center = [223, 295]
        center = [223, 256]
        x_min = center[0] - 64
        y_min = center[1] - 64
        ct_array = ct_array[:, x_min:x_min+128, y_min:y_min+128]
        gt_array = gt_array[:, x_min:x_min+128, y_min:y_min+128]

        ct_array = (ct_array - HU_lower) / (HU_upper - HU_lower) * 255

        # # save crop mask
        # for idx in range(ct_array.shape[0]):
        #     mask = gt_array[idx]
        #     mask[mask > 0] = 255
        #     cv.imwrite(join(mask_save_path_crop, str(idx) + '.png'), mask)
        # continue

        for idx in range(ct_array.shape[0]):
            ct_slice = ct_array[idx]
            cv.imwrite(join(gray_save_path, str(idx) + '.png'), ct_slice)

            gray_img = cv.imread(join(gray_save_path, str(idx) + '.png'), 0)
            gray_bgr = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

            label = gt_array[idx]
            label_list = np.unique(label)

            for label_id in label_list:
                if label_id != 0:
                    # print('slice:', idx)
                    label_temp = copy.deepcopy(label)
                    label_temp[label_temp != label_id] = 0
                    label_temp[label_temp == label_id] = 1

                    labels = measure.label(label_temp)
                    # print('pixels = ', labels.sum())
                    # regionprops中标签为0的区域会被自动忽略
                    for region in measure.regionprops(labels):
                        region_image = np.zeros(label_temp.shape)
                        # region_image[region.coords] = 1
                        region_image[region.coords[:, 0], region.coords[:, 1]] = 1

                        edges = feature.canny(region_image, low_threshold=1, high_threshold=1)
                        index = np.where(edges)
                        gray_bgr[index] = color_map[label_id - 1]

            cv.imwrite(join(ct_save_path_crop, str(idx) + '.png'), gray_bgr)

print('--------------------------------------------------\n')

