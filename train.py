from generator import *
from loss import YoloLoss
from data_augmentation import AllColorAugmentation, Cutout

tf.compat.v1.disable_eager_execution()


class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        if not training:
            return x
        return x * tf.math.tanh(tf.math.softplus(x))


def conv_block(x, kernel_size=3, filters=32, activation=True, strides=1):
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, padding="same", strides=strides,
                               kernel_initializer=tf.keras.initializers.HeNormal(),
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-6, l2=2e-5)
                               )(x)
    if activation:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        #x = tf.keras.layers.Mish()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def upsample_block(x, filters, size, stride=2):
    """
    x - the input of the upsample block
    filters - the number of filters to be applied
    size - the size of the filters
    """
    x = tf.keras.layers.Convolution2DTranspose(kernel_size=size, filters=filters, strides=stride, padding="same",
                                               kernel_initializer=tf.keras.initializers.HeNormal(),
                                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-6, l2=2e-5)
                                               )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), true_boxes_shape=(1, 1, 1, MAX_BOXES_PER_IMAGES, 4),
               no_classes=len(ENCODE_LABEL), no_anchors=3, alpha=1.0):
    inputs = tf.keras.layers.Input(shape=input_shape)
    true_boxes = tf.keras.layers.Input(shape=true_boxes_shape)
    x = AllColorAugmentation()(inputs)
    x = Cutout(8)(x)
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    mobilenet_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False,
                                                                  alpha=alpha)
    downsample_skip_layer_name = ["block_6_expand_relu",
                                  "block_10_expand_relu",
                                  # "block_13_expand_relu",
                                  "block_14_expand_relu"
                                  ]

    down_stack = tf.keras.Model(inputs=mobilenet_v2.input,
                                outputs=[mobilenet_v2.get_layer(name).output for name in downsample_skip_layer_name],
                                name="down_stack")
    down_stack.trainable = False

    skips = down_stack(x, training=False)
    x = skips[-1]

    for skip_layer in reversed(skips[:-1]):
        x = upsample_block(x, skip_layer.shape[-1], 3)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Concatenate()([x, skip_layer])

    x = conv_block(x, filters=192, strides=2)
    x = conv_block(x, filters=192)
    x = conv_block(x, filters=192)
    x = conv_block(x, filters=192)

    x = conv_block(x, filters=128)
    x = conv_block(x, filters=128)
    x = conv_block(x, filters=128)
    x = conv_block(x, filters=128)

    x = conv_block(x, filters=64, strides=2)
    x = conv_block(x, filters=64)
    x = conv_block(x, filters=64)
    x = conv_block(x, filters=64)

    x = conv_block(x, filters=32)
    x = conv_block(x, filters=32)
    x = conv_block(x, filters=32)
    x = conv_block(x, filters=32)
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=no_anchors * (4 + 1 + no_classes),
                               padding="same",
                               # strides=2, activation="relu"
                               kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.Reshape((GRID_SIZE, GRID_SIZE, no_anchors, 4 + 1 + no_classes), name="final_output")(x)
    output = tf.keras.layers.Lambda(lambda args: args[0], name="hack_layer")([x, true_boxes])
    return tf.keras.Model(inputs=[inputs, true_boxes], outputs=output, name="custom_yolo"), true_boxes


def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), true_boxes_shape=(1, 1, 1, MAX_BOXES_PER_IMAGES, 4),
                no_classes=len(ENCODE_LABEL), no_anchors=3, alpha=1.0):
    return build_unet(input_shape, true_boxes_shape, no_classes, no_anchors, alpha)


def plot_history(history):
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0, 1])
    plt.legend(loc='upper left')
    plt.show()


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
    def __init__(self, epochs=5, batch_size=BATCH_SIZE,
                 n_min=1e-5, n_max=4e-4, T=None,
                 path_to_model=None, alpha=1.0,
                 limit_batches=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_generator = DataGenerator(PATH_TO_TRAIN, self.batch_size, limit_batches=limit_batches)
        self.validation_generator = DataGenerator(PATH_TO_VALIDATION, self.batch_size, limit_batches=limit_batches)
        self.T = T if T is not None else 2 * batch_size
        self.n_min = n_min
        self.n_max = n_max
        self.alpha = alpha
        self.model = None
        self.true_boxes = None
        if path_to_model is None:
            self.new_model()
        else:
            self.load_model(path_to_model)

    def train(self, name="model.h5", fine_tune=False):
        if fine_tune:
            self.model.trainable = True
        self.model.summary()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=0.5,
                                                              # decay=0.0
                                                              ),
                           loss=YoloLoss(anchors=self.train_generator.anchors, true_boxes=self.true_boxes))

        history = self.model.fit(self.train_generator, validation_data=self.validation_generator, epochs=self.epochs,
                                 callbacks=[CosineAnnealingScheduler(self.n_min, self.n_max, self.T),
                                            tf.keras.callbacks.ModelCheckpoint(
                                                filepath=f"weights/{name}",
                                                save_best_only=True,
                                                verbose=2
                                            ),
                                            tf.keras.callbacks.TerminateOnNaN(),
                                            tf.keras.callbacks.EarlyStopping(patience=3, min_delta=1e-3, verbose=2),
                                            tf.keras.callbacks.TensorBoard(log_dir=f"info_about_runs/{name}")
                                            ]
                                 , workers=os.cpu_count())
        # tf.keras.models.save_model(self.model, f"weights/{name}")
        plot_history(history.history)

    def load_model(self, path: str):
        self.model, self.true_boxes = build_model(alpha=self.alpha)
        if "fine_tuned" in path:
            self.model.trainable = True
        self.model.load_weights(f"weights/{path}")

    def new_model(self):
        self.model, self.true_boxes = build_model(alpha=self.alpha)


def train():
    t = Train(epochs=2, n_min=1e-7, n_max=1e-5, path_to_model="model_v8_2.h5")
    t.train(name="model_v8_3.h5")


def fine_tune():
    fine_tune = Train(epochs=8, n_min=1e-9, n_max=1e-6, path_to_model="model_v4_4.h5")
    fine_tune.train(name="model_v4_4_fine_tuned.h5", fine_tune=True)


if __name__ == '__main__':
    # tf.keras.applications.mobilenet_v2.MobileNetV2().summary()
    # model, _ = build_model()
    # model.summary()
    train()
    # fine_tune()
