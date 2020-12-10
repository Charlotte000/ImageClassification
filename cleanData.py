import os
import cv2
from settings import IMAGE_SIZE

def clear_data():
    # Load ImageNet
    for name in os.listdir('.\\Data\\ImageNet\\'):
        try:
            file_path = f".\\Data\\ImageNet\\{name}"
            cv2.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        except Exception:
            os.remove(file_path)
            print(f"Bad image: {file_path}")

    # Load Stanford
    for name in ['mug', 'forks', 'keyboard', 'scissors', 'stapler']:
        for i in range(0, 200):
            index = str(i).zfill(3)
            try:
                file_path = f".\\Data\\Stanford\\{name}\\{name}.{index}.jpg"
                cv2.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))

            except Exception:
                os.remove(file_path)
                print(f"Bad image: {file_path}")

clear_data()
