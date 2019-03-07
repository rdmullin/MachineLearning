# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import os
import cv2

Dataset_path = "animals/"
data = []

#Target variable
label = []

#Find subfolders of animals
class_folders = os.listdir(Dataset_path)

#Remove hidden folders from class_folders
for folderName in class_folders:
    if folderName.startswith('.'):
        class_folders.remove(folderName)

#Load images into data[] array and resize them to 32x32 pixels RGB
for class_name in class_folders:
    image_list = os.listdir(Dataset_path + class_name)
    for image_name in image_list:
        image = cv2.imread(Dataset_path + class_name + '/' + image_name)
        image = cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
        
        data.append(image)
        label.append(class_name)

#Turn each 32x32x3 RGG into a single dimension 3072 column vector
Feature_flat_data = []
for d in data:
    Feature_flat_data.append(d.ravel())


#Import LabelEncoder
from sklearn import preprocessing
#creates labelEncoder
le = preprocessing.LabelEncoder()
#Convert string labels into numbers
label_encoded = le.fit_transform(label)

#Split Data into training and testing data

from sklearn.model_selection import train_test_split

(trainX,testX,trainY,testY) = train_test_split(Feature_flat_data,label_encoded,
    test_size = 0.25,random_state = 42)


#Import Model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 3)


#Fit model
model.fit(trainX,trainY)

from sklearn.metrics import classification_report

print(classification_report(testY,model.predict(testX), target_names = le.classes_))


#Cross-validation
from sklearn.model_selection import cross_val_score



k_range = range(1,10)
k_scores = []

'''
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,Feature_flat_data,label_encoded,cv=10,scoring = 'accuracy')
    k_scores.append(scores.mean())
'''
print("Done")








