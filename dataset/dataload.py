#encoding=UTF-8

import os
import random

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
import numpy as np
import cv2 as cv

class Dataset(dataset):
    def __init__(self, cts, gts, random_crop=False, crop_size=[128, 128]):
        self.cts = cts
        self.gts = gts
        self.crop_size = crop_size
        self.random_crop = random_crop

    def __getitem__(self, index):
        ct_array = self.cts[index, ::]
        seg_array = self.gts[index, ::]

        center = [223, 256]

        if self.random_crop:
            x_min = random.randint(center[0] - self.crop_size[0] // 2 - 4, center[0] - self.crop_size[1] // 2 + 4)
            y_min = random.randint(center[1] - self.crop_size[0] // 2 - 4, center[1] - self.crop_size[1] // 2 + 4)
        else:
            x_min = center[0] - self.crop_size[0] // 2
            y_min = center[1] - self.crop_size[0] // 2

        new_ct_array  = torch.FloatTensor(ct_array[x_min:x_min+self.crop_size[0], y_min:y_min+self.crop_size[1]]).unsqueeze(dim=0)
        new_seg_array = torch.FloatTensor(seg_array[x_min:x_min+self.crop_size[0], y_min:y_min+self.crop_size[1]])

        return new_ct_array, new_seg_array

    def __len__(self):
        return len(self.cts)

class Dataset_location(dataset):
    def __init__(self, cts, gts):
        self.cts = cts
        self.gts = gts

    def __getitem__(self, index):
        ct_array = self.cts[index, ::]
        seg_array = self.gts[index, ::]

        new_ct_array  = torch.FloatTensor(ct_array).unsqueeze(dim=0)
        new_seg_array = torch.FloatTensor(seg_array)

        return new_ct_array, new_seg_array

    def __len__(self):
        return len(self.cts)

