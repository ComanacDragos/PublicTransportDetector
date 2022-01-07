import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf
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
    def __init__(self, crop_size, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size  # cropped region will be cropSize*2+1

    def call(self, x, training=None):
        if not training:
            return x
        return tf.map_fn(lambda elem: cutout(elem, self.crop_size), x)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'crop_size': self.crop_size
        })
        return config


class RandomHue(tf.keras.layers.Layer):
    def __init__(self, delta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, x, training=None):
        if not training:
            return x
        return tf.image.random_hue(x, self.delta)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'delta': self.delta
        })
        return config


class RandomSaturation(tf.keras.layers.Layer):
    def __init__(self, lower=5, upper=10, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, x, training=None):
        if not training:
            return x
        return tf.image.random_saturation(x, self.lower, self.upper)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lower': self.lower,
            'upper': self.upper
        })
        return config


class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, delta=0.3, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, x, training=None):
        if not training:
            return x
        return tf.image.random_brightness(x, self.delta)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'delta': self.delta
        })
        return config


class RandomContrast(tf.keras.layers.Layer):
    def __init__(self, lower=1, upper=2, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, x, training=None):
        if not training:
            return x
        return tf.image.random_contrast(x, self.lower, self.upper)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lower': self.lower,
            'upper': self.upper
        })
        return config


class RandomColorAugmentation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            RandomHue(),
            RandomSaturation(),
            RandomBrightness(),
            RandomContrast(),
        ]

    def call(self, x, training=None):
        if not training:
            return x
        return self.layers[np.random.randint(0, len(self.layers))].call(x, training)


class AllColorAugmentation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            RandomHue(),
            RandomSaturation(),
            RandomBrightness(),
            RandomContrast(),
        ]

    def call(self, x, training=None):
        if not training:
            return x
        for layer in self.layers:
            x = layer.call(x, training)
        return x


if __name__ == '__main__':
    img = Image(PATH_TO_VALIDATION, "4a23eee283f294b6.jpg").image
    img = np.stack([img] * 2)
    cutout_layer = Cutout(50)
    hue_layer = RandomHue()
    saturation_layer = RandomSaturation()
    brightness_layer = RandomBrightness()
    contrast_layer = RandomContrast()
    all_layer = AllColorAugmentation()
    random_layer = RandomColorAugmentation()

    rows = 4
    cols = 2

    fig, axs = plt.subplots(rows, cols, figsize=(30, 30))
    plt.subplot(rows, cols, 1)
    plt.imshow(img[0])
    plt.title("original")

    plt.subplot(rows, cols, 2)
    plt.imshow(cutout_layer.call(img, training=True)[0])
    plt.title("cutout")

    plt.subplot(rows, cols, 3)
    plt.imshow(hue_layer.call(img, training=True)[0])
    plt.title("hue")

    plt.subplot(rows, cols, 4)
    plt.imshow(saturation_layer.call(img, training=True)[0])
    plt.title("saturation")

    plt.subplot(rows, cols, 5)
    plt.imshow(brightness_layer.call(img, training=True)[0])
    plt.title("brightness")

    plt.subplot(rows, cols, 6)
    plt.imshow(contrast_layer.call(img, training=True)[0])
    plt.title("contrast")

    plt.subplot(rows, cols, 7)
    plt.imshow(random_layer.call(img, training=True)[0])
    plt.title("random")

    plt.subplot(rows, cols, 8)
    plt.imshow(all_layer.call(img, training=True)[0])
    plt.title("all")

    plt.tight_layout()
    plt.show()
