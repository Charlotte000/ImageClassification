from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

import cv2
import numpy as np
import os
from random import shuffle

from settings import IMAGE_SIZE

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    x, y = 15, 15
    w, h = IMAGE_SIZE - 30, IMAGE_SIZE - 30
    new_image = cv2.warpAffine(image, rot_mat, (col,row))[y:y+h, x:x+w]
    return cv2.resize(new_image, (IMAGE_SIZE, IMAGE_SIZE))

def load_training_data():
    data = []

    # Load ImageNet
    files = os.listdir(os.path.join(".", "Data", "ImageNet"))
    for name in files[:round(len(files) * .9)]:
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
        for i in range(0, 180):
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

def create_model():
    training_data = load_training_data()
    print(f'Loaded {len(training_data)} images')
    x = []
    y = []
    for img, label in training_data:
        x.append(img)
        y.append(label)

    x = np.array(x).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    x = x / 255.0
    y = np.array(y)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, batch_size=32, validation_split=.1, epochs=3)
    model.save('model')

create_model()
