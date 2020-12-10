import os
import cv2
from settings import IMAGE_SIZE

def clear_data():
    ''' Deletes corrupted files '''
    # ImageNet
    for name in os.listdir(os.path.join(".", "Data", "ImageNet")):
        try:
            path = os.path.join(".", "Data", "ImageNet", name)
            cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        except Exception:
            os.remove(path)
            print(f"Bad image: {path}")

    # Stanford
    for name in ['mug', 'forks', 'keyboard', 'scissors', 'stapler']:
        for i in range(0, 200):
            index = str(i).zfill(3)
            try:
                path = os.path.join(".", "Data", "Stanford", name, f"{name}.{index}.jpg")
                cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))

            except Exception:
                os.remove(path)
                print(f"Bad image: {path}")

clear_data()
