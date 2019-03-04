# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import cv2
import numpy as np

Dataset_path = "animals/"
data = []

class_folders = os.listdir(Dataset_path)
#Remove hidden folders from class_folders
for folderName in class_folders:
    if folderName.startswith('.'):
        class_folders.remove(folderName)

for class_name in class_folders:
    image_list = os.listdir(Dataset_path + class_name)
    for image_name in image_list:
        image = cv2.imread(Dataset_path + class_name + '/' + image_name)
        image = cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
        data.append(image)

mydata = np.array(data)



