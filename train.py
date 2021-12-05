import numpy as np

from generator import *


def conv_layer(x, kernel_size=3, filters=32):
    conv = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, padding="same", activation="relu",
                                  kernel_initializer=tf.keras.initializers.HeNormal())(x)
    bn = tf.keras.layers.BatchNormalization()(conv)
    return tf.keras.layers.ReLU()(bn)


def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), trainable=False):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    mobilenet_v2.trainable = trainable
    x = mobilenet_v2(x, training=trainable)
    x = tf.keras.layers.ReLU()(x)
    # x = conv_layer(x, filters=640)
    x = conv_layer(x, filters=320)
    # x = conv_layer(x, filters=160)
    x = conv_layer(x, filters=80)
    x = conv_layer(x, filters=40)
    x = tf.reshape(x, (-1, 13, 13, 5, 8))
    x = tf.cast(x, tf.float64)
    output = tf.where([True, True, False, False, True, False, False, False], tf.sigmoid(x), x)
    return tf.keras.Model(inputs=inputs, outputs=output, name="custom_yolo")


def plot_history(history):
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0, 1])
    plt.legend(loc='upper left')


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


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, l_coord=5, l_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def loss_for_one(self, y_true, y_pred):
        """
                Each anchor is composed of 8 values:
                0, 1: x, y position
                2, 3: width, height
                4: if there is an object
                5, 6, 7: probabilities
                """
        objectness_scores, objectness_scores_hat = y_true[:, :, :, 4], y_pred[:, :, :, 4]
        x, x_hat = y_true[:, :, :, 0], y_pred[:, :, :, 0]
        y, y_hat = y_true[:, :, :, 1], y_pred[:, :, :, 1]
        w, w_hat = tf.sqrt(y_true[:, :, :, 2]), tf.sqrt(y_pred[:, :, :, 2])
        h, h_hat = tf.sqrt(y_true[:, :, :, 3]), tf.sqrt(y_pred[:, :, :, 3])
        class_scores, class_scores_hat = y_true[:, :, :, 5:], y_pred[:, :, :, 5:]
        class_scores_hat = tf.keras.backend.softmax(class_scores_hat)

        xy_loss = self.l_coord * tf.reduce_sum(objectness_scores * ((x - x_hat) ** 2 + (y - y_hat) ** 2))
        wh_loss = self.l_coord * tf.reduce_sum(objectness_scores * ((w - w_hat) ** 2 + (h - h_hat) ** 2))
        obj_diff = (objectness_scores - objectness_scores_hat) ** 2
        obj_loss = tf.reduce_sum(objectness_scores * obj_diff) + \
                   self.l_noobj * tf.reduce_sum((1 - objectness_scores) * obj_diff)

        class_loss = tf.reduce_sum(tf.stack([objectness_scores] * 3, axis=-1) * ((class_scores - class_scores_hat) ** 2))
        return xy_loss + wh_loss + obj_loss + class_loss

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        pass


class Train:
    def __init__(self, epochs=5, batch_size=32, n_min=1e-5, n_max=4e-4, T=None, path_to_model=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_generator = DataGenerator(PATH_TO_TRAIN, self.batch_size)
        self.validation_generator = DataGenerator(PATH_TO_VALIDATION, self.batch_size)
        self.T = T if T is not None else 2 * batch_size
        self.n_min = n_min
        self.n_max = n_max
        self.model = None
        if path_to_model is None:
            self.new_model()
        else:
            self.load_model(path_to_model)

    def train(self, name="model.h5", fine_tune=False):
        if fine_tune:
            self.model.trainable = fine_tune
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=YoloLoss(),
                      # metrics=[],
                      )

        history = self.model.fit(self.train_generator, validation_data=self.validation_generator, epochs=self.epochs,
                            callbacks=[CosineAnnealingScheduler(self.n_min, self.n_max, self.T),
                                       tf.keras.callbacks.ModelCheckpoint(
                                           filepath="weights/checkpoint/checkpoint",
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=2
                                       ),
                                       tf.keras.callbacks.TerminateOnNaN(),
                                       tf.keras.callbacks.EarlyStopping(patience=3, min_delta=1e-3, verbose=2)
                                       ], workers=os.cpu_count())
        self.model.save(f"weights/{name}")
        plot_history(history.history)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def new_model(self):
        self.model = build_model()


if __name__ == '__main__':
    #Train(epochs=1).train()
    model = build_model()
    #model.summary()
    validation_generator = DataGenerator(PATH_TO_VALIDATION, 32, shuffle=False)
    images, y_true = validation_generator[1]
    y_pred = model.predict(images)
    loss = YoloLoss()
    print(y_pred.shape)

    print(loss.loss_for_one(y_true[0], y_pred[0]))


    """
     a = np.arange(2 * 2 * 5 * 8)
    a = tf.convert_to_tensor(a)
    a = tf.reshape(a, (2,2,5,8))
    b = a[:, :, :, 4]
    print(b)
    c = tf.stack([b]*3,axis=-1)
    print(c)
    a = np.arange(2 * 2 * 5 * 8)
    a = tf.convert_to_tensor(a)
    print(a)
    a = tf.reshape(a, (2,2,5,8))
    print(a)
    a = tf.cast(a, tf.float64)
    a = tf.where([True, True, False, False, True, True, True, True], tf.sigmoid(a), a)
    print(a)
    
    a = tf.convert_to_tensor(a)
    a = tf.reshape(a, (2,2,5,8))
    print(a)
    print(a[0, 0, 0, :])
    a = tf.convert_to_tensor(a)
    a = tf.reshape(a, (2, 2, 5, 8))
    print(a)
    a = tf.cast(a, tf.float64)
    print(tf.keras.activations.softmax(a[:, :, :, 5:]))

    print(tf.reduce_sum(tf.keras.activations.softmax(a[:, :, :, 5:]), axis=-1))

    """
