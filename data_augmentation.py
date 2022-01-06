import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf
from generator import DataGenerator
from image import Image
from settings import *


def clip(x, left, right):
    return max(left, min(x, right))


def cutout(x, cropSize):
    cut_x = x
    shape = x.get_shape()

    mask = np.ones(shape)
    x_coord = np.random.randint(0, shape[0])
    y_coord = np.random.randint(0, shape[1])
    tl_x = clip(x_coord - cropSize, 0, shape[0])
    tl_y = clip(y_coord - cropSize, 0, shape[1])
    br_x = clip(x_coord + cropSize, 0, shape[0])
    br_y = clip(y_coord + cropSize, 0, shape[1])
    mask[tl_x:br_x, tl_y:br_y, :] = np.zeros((br_x - tl_x, br_y - tl_y, shape[2]))
    cut_x = tf.where(tf.convert_to_tensor(mask, dtype=tf.bool), cut_x, 0)
    return cut_x


class Cutout(tf.keras.layers.Layer):
    def __init__(self, cropSize, **kwargs):
        super().__init__(**kwargs)
        self.cropSize = cropSize  # cropped region will be cropSize*2+1

    def call(self, x, training=None):
        if not training:
            return x
        return tf.map_fn(lambda elem: cutout(elem, self.cropSize), x)


if __name__ == '__main__':
    img = np.expand_dims(Image(PATH_TO_VALIDATION, "4a23eee283f294b6.jpg").image, 0)
    cutout_layer = Cutout(50)

    fig, axs = plt.subplots(2, 1, figsize=(30, 30))
    plt.subplot(2, 1, 1)
    plt.imshow(img[0])
    plt.subplot(2, 1, 2)
    plt.imshow(cutout_layer.call(img, training=True)[0])

    plt.tight_layout()
    plt.show()