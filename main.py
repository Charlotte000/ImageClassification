import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from tensorflow import keras
import cv2

from settings import IMAGE_SIZE

def load_model():
    return keras.models.load_model('model')

def loadImage(name):
    img = cv2.resize(cv2.imread(name, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
    return img

model = load_model()
path = sys.argv[1]
img = loadImage(path)
print(model.predict([img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)])[0][0])
