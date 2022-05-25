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
    [0, 255, 0],   # Green            --chiasm
    [0, 0, 255],   # Red              --pituitary
    [0, 165, 255], # Orange           --nerve L
    [255, 0, 0],   # Blue             --nerve R
]

HU_upper = 300
HU_lower = -100

ct_path_4organ = '/home/zjm/Data/HaN_OAR_raw/visualization/'
# pred_path = '/home/zjm/Project/segmentation/BLSC_2D/outputs_without_ensemble/val_subset5_UNet_newDiceFlood0.003_128_512_512NumofOrgan_4/pred_label_rm_3d/'
pred_path = '/home/zjm/Project/segmentation/BLSC_2D/outputs/val_subset5_UNet_[128, 128]_celoss_expand0NumofOrgan_4/Exp3/pred_label/'
save_path = join(pred_path.split('pred_label')[0], 'visual_crop')

if not os.path.exists(save_path):
    os.makedirs(save_path)

center = [223, 256]
x_min = center[0] - 64
y_min = center[1] - 64


for root, dirs, files in os.walk(ct_path_4organ):
    # if 'ct_4organ' in root and files[0].endswith('.png'):
    if 'gray' in root and files[0].endswith('.png'):
        p_name = root.split('/')[-2]
        if int(p_name) > 40:
            print(root)
            p_save_path = join(save_path, str(p_name))
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)

            gt_img = sitk.ReadImage(join(pred_path, p_name + '.nii'))
            gt_array = sitk.GetArrayFromImage(gt_img)
            gt_array = gt_array[::, x_min:x_min + 128, y_min:y_min + 128]
            print(np.unique(gt_array))

            for idx in range(len(files)):
                gray_img = cv.imread(join(root, str(idx) + '.png'), 0)
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
                            gray_bgr[index] = color_map[int(label_id - 1)]

                cv.imwrite(join(p_save_path, str(idx) + '.png'), gray_bgr)


print('--------------------------------------------------\n')

