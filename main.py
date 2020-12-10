import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

from settings import IMAGE_SIZE

def load_model():
    return keras.models.load_model('model')

def loadImage(name):
    img = cv2.resize(cv2.imread(name, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
    return img


model = load_model()

for name in os.listdir(os.path.join(".", "Tests")):
    img = loadImage(os.path.join(".", "Tests", name))
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title(str(model.predict([img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)])[0][0]))
plt.show()