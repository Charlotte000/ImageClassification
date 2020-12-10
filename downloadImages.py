from threading import Thread
from os.path import join
import requests

def load_image_net():
    ''' Load dataset from http://image-net.org/synset?wnid=n03797390 '''
    url = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03797390'

    def _load_part(urls):
        for url in urls:
            try:
                name = url.split('/')[-1]
                data = requests.get(url).content
                path = join(".", "Data", "ImageNet", name)
                with open(path, 'wb') as write_file:
                    write_file.write(data)
            except:
                print(f'Bad url: {url}')

    image_list = requests.get(url).content.decode().split('\r\n')
    for i in range(0, len(image_list), 10):
        Thread(target=_load_part, args=(image_list[i:i+10], )).start()

def load_stanford():
    ''' Load dataset from http://ai.stanford.edu/ '''
    url = 'http://ai.stanford.edu/~asaxena/robotdatacollection/real/'

    def _load_part(name, indexA, indexB):
        for i in range(indexA, indexB):
            try:
                index = str(i).zfill(3)
                data = requests.get(url + f'{name}/{name}.{index}.jpg').content
                path = join(".", "Data", "Stanford", name, f"{name}.{index}.jpg")
                with open(path, 'wb') as write_file:
                    write_file.write(data)
            except:
                print(f"Bad url: {url}{name}/{name}.{index}.jpg")
    for name in ['mug', 'forks', 'keyboard', 'scissors', 'stapler', 'hammers', 'flipphones', 'pliers', 'telephone', 'watches']:
        for i in range(0, 200, 10):
            Thread(target=_load_part, args=(name, i, i + 10)).start()
