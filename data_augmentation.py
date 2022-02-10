import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from image import Image
from settings import *
from preprocessor import resize_image


@tf.function
def cutout(x, crop_size):
    image_shape = tf.convert_to_tensor((IMAGE_SIZE, IMAGE_SIZE))

    x_coord = tf.random.uniform(shape=[], maxval=IMAGE_SIZE, dtype=tf.int32)
    y_coord = tf.random.uniform(shape=[], maxval=IMAGE_SIZE, dtype=tf.int32)
    tl_x = tf.keras.backend.clip(tf.math.subtract(x_coord, crop_size), 0, IMAGE_SIZE - 1)
    tl_y = tf.keras.backend.clip(tf.math.subtract(y_coord, crop_size), 0, IMAGE_SIZE - 1)
    br_x = tf.keras.backend.clip(tf.math.add(x_coord, crop_size), 0, IMAGE_SIZE - 1)
    br_y = tf.keras.backend.clip(tf.math.add(y_coord, crop_size), 0, IMAGE_SIZE - 1)

    y_range = tf.range(tl_y, br_y, dtype=tf.int32) * IMAGE_SIZE
    x_range = tf.range(tl_x, br_x, dtype=tf.int32)

    values = tf.ones(shape=tf.math.multiply(tf.shape(x_range), tf.shape(y_range)))

    y_range = tf.reshape(y_range, (-1, 1))
    x_range = tf.reshape(x_range, shape=(1, -1))

    indices = tf.math.add(x_range, y_range)
    indices = tf.reshape(indices, shape=[-1])
    indices = tf.unravel_index(indices, dims=image_shape)
    indices = tf.transpose(indices)
    indices = tf.cast(indices, tf.int64)

    st = tf.SparseTensor(indices, values, tf.cast(image_shape, dtype=tf.int64))
    st_ordered = tf.compat.v1.sparse_reorder(st)
    mask = tf.compat.v1.sparse_tensor_to_dense(st_ordered)
    mask = tf.cast(mask, tf.bool)
    mask = tf.expand_dims(mask, -1)

    cut_x = tf.where(mask, 0, x)
    return cut_x


class Cutout(tf.keras.layers.Layer):
    def __init__(self, crop_size, **kwargs):
        """
        :param crop_size: half the size of the crop
        """
        super().__init__(**kwargs)
        self.crop_size = crop_size

    def call(self, x, training=None):
        x = tf.cast(x, dtype=tf.int32)
        if training:
            return tf.map_fn(lambda elem: cutout(elem, self.crop_size), x)
        return tf.keras.backend.in_train_phase(tf.map_fn(lambda elem: cutout(elem, self.crop_size), x), x)

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
        if training:
            return tf.image.random_hue(x, self.delta)
        return tf.keras.backend.in_train_phase(tf.image.random_hue(x, self.delta), x)

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
        if training:
            return tf.image.random_saturation(x, self.lower, self.upper)
        return tf.keras.backend.in_train_phase(tf.image.random_saturation(x, self.lower, self.upper), x)

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
        if training:
            return tf.image.random_brightness(x, self.delta)
        return tf.keras.backend.in_train_phase(tf.image.random_brightness(x, self.delta), x)

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
        if training:
            return tf.image.random_contrast(x, self.lower, self.upper)
        return tf.keras.backend.in_train_phase(tf.image.random_contrast(x, self.lower, self.upper), x)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lower': self.lower,
            'upper': self.upper
        })
        return config


@tf.function
def apply_transformation(layers, x, training):
    transformation = tf.random.uniform(shape=[], maxval=len(layers), dtype=tf.int32)
    if tf.equal(transformation, tf.constant(0)):
        return layers[0].call(x, training)
    elif tf.equal(transformation, tf.constant(1)):
        return layers[1].call(x, training)
    elif tf.equal(transformation, tf.constant(2)):
        return layers[2].call(x, training)
    else:
        return layers[3].call(x, training)


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
        if training:
            return apply_transformation(self.layers, x, training)
        return tf.keras.backend.in_train_phase(apply_transformation(self.layers, x, True), x)


def mosaic(images, min_size=50):
    x_cut = np.random.randint(min_size, IMAGE_SIZE - min_size)
    y_cut = np.random.randint(min_size, IMAGE_SIZE - min_size)
    resize_image(images[0], h=y_cut, w=x_cut)
    resize_image(images[1], h=y_cut, w=IMAGE_SIZE - x_cut)
    resize_image(images[2], h=IMAGE_SIZE - y_cut, w=x_cut)
    resize_image(images[3], h=IMAGE_SIZE - y_cut, w=IMAGE_SIZE - x_cut)

    x_grid = [
        [0, x_cut],
        [0, x_cut]
    ]
    y_grid = [
        [0, 0],
        [y_cut, y_cut]
    ]
    new_img = Image()
    new_img.image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.int)
    for i, img in enumerate(images):
        line = i // 2
        col = i % 2
        img.shift_boxes(x_grid[line][col], y_grid[line][col])
        new_img.bounding_boxes += img.bounding_boxes
    new_img.image[0:y_cut, 0:x_cut, :] = images[0].image
    new_img.image[0:y_cut, x_cut:IMAGE_SIZE, :] = images[1].image
    new_img.image[y_cut:IMAGE_SIZE, 0:x_cut, :] = images[2].image
    new_img.image[y_cut:IMAGE_SIZE, x_cut:IMAGE_SIZE, :] = images[3].image

    return new_img


def visualize_augmentations():
    img = Image(PATH_TO_VALIDATION, "4a23eee283f294b6.jpg").image
    img = np.stack([img] * 2)
    cutout_layer = Cutout(32)
    hue_layer = RandomHue()
    saturation_layer = RandomSaturation()
    brightness_layer = RandomBrightness()
    contrast_layer = RandomContrast()
    random_layer = RandomColorAugmentation()

    rows = 4
    cols = 2

    fig, axs = plt.subplots(rows, cols, figsize=(40, 40))
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
    plt.imshow(random_layer.call(img, training=True)[0])
    plt.title("random")

    plt.tight_layout()
    plt.savefig("documentation/augmentation.jpg")


if __name__ == '__main__':
    images = [
        Image(PATH_TO_VALIDATION, "0f4bfc46402a9f52.jpg"),
        Image(PATH_TO_VALIDATION, "3f10138eb086f2e9.jpg"),
        Image(PATH_TO_VALIDATION, "3fc5aee11ddb3651.jpg"),
        Image(PATH_TO_VALIDATION, "4a23eee283f294b6.jpg"),
    ]
    img = mosaic(images)
    plt.imshow(img.with_bboxes(3))
    plt.show()
    print(img.image.shape)
    # visualize_augmentations()
