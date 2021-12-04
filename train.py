import numpy as np

from generator import *


def conv_layer(kernel_size=3, filters=32):
    return tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters)


def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=13 * 13 * 5 * 8, trainable=False):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    mobilenet_v2.trainable = trainable
    x = mobilenet_v2(x, training=trainable)
    x = tf.keras.layers.ReLU()(x)

    # x = tf.reshape(x, (13, 13, 5, 8))
    # x = tf.cast(x, tf.float64)
    output = x  # tf.where([True, True, False, False, True, True, True, True], tf.sigmoid(x), x)
    return tf.keras.Model(inputs=inputs, outputs=output, name="custom_yolo")


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self, n_min, n_max, T):
        super(CosineAnnealingScheduler, self).__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.T = T

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        scheduled_lr = self.n_min + (1 / 2) * (self.n_max - self.n_min) * (1 + np.cos(epoch / self.T * np.pi))
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print(f"\nEpoch {epoch + 1}: Learning rate is {scheduled_lr}.")


class Train:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_generator = DataGenerator(PATH_TO_TRAIN, self.batch_size)
        self.validation_generator = DataGenerator(PATH_TO_VALIDATION, self.batch_size)


if __name__ == '__main__':
    model = build_model()
    model.summary()

    a = np.arange(2 * 2 * 5 * 8)
    """
    a = tf.convert_to_tensor(a)
    print(a)
    a = tf.reshape(a, (2,2,5,8))
    print(a)
    a = tf.cast(a, tf.float64)
    a = tf.where([True, True, False, False, True, True, True, True], tf.sigmoid(a), a)
    print(a)
    """
