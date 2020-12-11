import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import cv2

from settings import IMAGE_SIZE

def load_model():
    return keras.models.load_model('model')

def loadImage(name):
    img = cv2.resize(cv2.imread(name, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
    return img


model = load_model()

TP = FP = TN = FN = 0

for name in os.listdir(os.path.join(".", "Tests")):
    img = loadImage(os.path.join(".", "Tests", name))
    actual_label = int(name.startswith("1"))
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
os.system("pause")
