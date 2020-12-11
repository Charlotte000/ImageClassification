import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import cv2
from random import shuffle
import numpy as np

from settings import IMAGE_SIZE

def load_model():
    return keras.models.load_model('model')

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    x, y = 15, 15
    w, h = IMAGE_SIZE - 30, IMAGE_SIZE - 30
    new_image = cv2.warpAffine(image, rot_mat, (col,row))[y:y+h, x:x+w]
    return cv2.resize(new_image, (IMAGE_SIZE, IMAGE_SIZE))

def load_testing_data():
    data = []

    # Load ImageNet
    files = os.listdir(os.path.join(".", "Data", "ImageNet"))
    for name in files[round(len(files) * .9):]:
        try:
            path = os.path.join(".", "Data", "ImageNet", name)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
            data.append([img, 1])
            data.append([rotateImage(img, -10), 1])
            data.append([rotateImage(img, 10), 1])
        except Exception:
            print(f"Bad image: {name}")

    # Load Stanford
    for name in ['mug', 'forks', 'keyboard', 'scissors', 'stapler', 'hammers', 'flipphones', 'pliers', 'telephone', 'watches']:
        for i in range(180, 200):
            index = str(i).zfill(3)
            try:
                path = os.path.join(".", "Data", "Stanford", name, f"{name}.{index}.jpg")
                img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
                data.append([img, int(name == 'mug')])
                data.append([rotateImage(img, 10), int(name == 'mug')])
                data.append([rotateImage(img, -10), int(name == 'mug')])
            except Exception:
                print(f"Bad image: {name}.{index}")

    shuffle(data)
    return data


model = load_model()

TP = FP = TN = FN = 0

data = load_testing_data()
print(f"Loaded {len(data)} files")
for img, actual_label in data:
    predicted_label = int(model.predict([img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)])[0][0])

    if actual_label == predicted_label:
        if actual_label == 1:
            TP += 1 # True Positive
        else:
            TN += 1 # True Negative
    else:
        if predicted_label == 1:
            FP += 1 # False Positive
        else:
            FN += 1 # False Negative

print(f"TP: {TP}\nFP: {FP}\nTN: {TN}\nFN: {FN}")

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(f"Precision: {round(precision, 4)}")
print(f"Recall: {round(recall, 4)}")
print(f"F measure: {round(f1, 4)}")
