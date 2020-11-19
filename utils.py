import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2

def load_image(path):
    img= np.array(cv2.imread(path, cv2.IMREAD_ANYDEPTH))
    return img #np.clip(img, 0, 9000.0)


def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
