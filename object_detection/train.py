from data_augmentation import Cutout, RandomColorAugmentation
from generator import *
from loss import YoloLoss
from tensorflow.keras.layers import Input, Concatenate, Conv2D, DepthwiseConv2D, Add, LeakyReLU, BatchNormalization, \
    Convolution2DTranspose, Reshape, Lambda, Dropout
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard, EarlyStopping, ReduceLROnPlateau

from tensorflow.python.keras.callbacks import History

L1 = 2e-6
L2 = 2e-5
LEAKY_RELU_ALPHA = 0.1


class BlockCreator:
    """
    Facade for creating types of neural network blocks
    """
    @staticmethod
    def conv_block(inputs, kernel_size=3, filters=32, activation=True, add_skip_connection=True, strides=1):
        x = Conv2D(kernel_size=kernel_size, filters=filters, padding="same", strides=strides,
                   kernel_initializer=HeNormal(),
                   kernel_regularizer=l1_l2(l1=L1, l2=L2)
                   )(inputs)
        if add_skip_connection:
            x = Conv2D(kernel_size=kernel_size, filters=filters, padding="same", strides=strides,
                       kernel_initializer=HeNormal(),
                       kernel_regularizer=l1_l2(l1=L1, l2=L2)
                       )(x)
            x = Add()([inputs, x])
        if activation:
            x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
        x = BatchNormalization()(x)
        return x

    @staticmethod
    def upsample_block(x, filters, kernel_size=3, strides=2):
        x = Convolution2DTranspose(kernel_size=kernel_size, filters=filters, strides=strides,
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.HeNormal(),
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2)
                                   )(x)
        x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
        x = BatchNormalization()(x)
        return x

    @staticmethod
    def inverted_residual_block(inputs, expand, squeeze, add_skip_connection=True):
        x = Conv2D(expand, (1, 1),
                   kernel_initializer=tf.keras.initializers.HeNormal(),
                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2)
                   )(inputs)
        x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
        x = BatchNormalization()(x)
        x = DepthwiseConv2D((3, 3),
                            padding="same",
                            depthwise_initializer=tf.keras.initializers.HeNormal(),
                            depthwise_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2))(x)
        x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
        x = BatchNormalization()(x)
        x = Conv2D(squeeze, (1, 1),
                   kernel_initializer=tf.keras.initializers.HeNormal(),
                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2))(x)
        x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
        x = BatchNormalization()(x)
        if add_skip_connection:
            x = Add()([x, inputs])
        return x


def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), true_boxes_shape=(1, 1, 1, MAX_BOXES_PER_IMAGES, 4),
                no_classes=len(ENCODE_LABEL), no_anchors=NO_ANCHORS, alpha=1.0, inference_only=False):
    if not inference_only:
        tf.compat.v1.disable_eager_execution()

    inputs = Input(shape=input_shape)
    true_boxes = Input(shape=true_boxes_shape)
    x = inputs
    if not inference_only:
        x = RandomColorAugmentation()(x)
        x = Cutout(192)(x)
        x = tf.cast(x, tf.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    mobilenet_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False,
                                                                  alpha=alpha)
    downsample_skip_layer_name = [
        # "block_4_expand_relu",
        # "block_6_expand_relu",
        # "block_8_expand_relu",
        # "block_10_expand_relu",
        # "block_14_expand_relu",
        "block_7_add",
        "block_9_add",
        "block_12_add",
        "block_15_add"
    ]

    down_stack = tf.keras.Model(inputs=mobilenet_v2.input,
                                outputs=[mobilenet_v2.get_layer(name).output for name in downsample_skip_layer_name],
                                name="down_stack")
    down_stack.trainable = False

    skips = down_stack(x, training=False)
    x = skips[-1]

    for skip_layer in reversed(skips[:-1]):
        x = BlockCreator.upsample_block(x, skip_layer.shape[-1], strides=int(skip_layer.shape[1] / x.shape[1]))
        x = Concatenate()([x, skip_layer])

    x = Dropout(0.3)(x)

    x = BlockCreator.conv_block(x, filters=128, strides=2, add_skip_connection=False)

    x = BlockCreator.inverted_residual_block(x, 512, 128)
    x = BlockCreator.inverted_residual_block(x, 512, 128)

    x = BlockCreator.conv_block(x, filters=no_anchors * (4 + 1 + no_classes),
                                add_skip_connection=False, activation=False)
    x = Reshape((GRID_SIZE, GRID_SIZE, no_anchors, 4 + 1 + no_classes), name="final_output")(x)
    if inference_only:
        return tf.keras.Model(inputs=inputs, outputs=x)
    output = Lambda(lambda args: args[0], name="hack_layer")([x, true_boxes])
    return tf.keras.Model(inputs=[inputs, true_boxes], outputs=output, name="custom_yolo"), true_boxes


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
        self.learning_rates = {
            # 0: 1e-3
        }

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch not in self.learning_rates:
            scheduled_lr = self.n_min + (1 / 2) * (self.n_max - self.n_min) * (1 + np.cos(epoch / self.T * np.pi))
        else:
            scheduled_lr = self.learning_rates[epoch]
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print(f"\nEpoch {epoch + 1}: Learning rate is {scheduled_lr}.")


class Trainer:
    def train(self, name: str) -> History: pass

    def load_model(self, path_to_model: str): pass

    @property
    def model(self) -> tf.keras.Model:
        return None


class SimpleTrainer(Trainer):
    def __init__(self, epochs=5, batch_size=BATCH_SIZE,
                 n_min=1e-5, n_max=4e-4, T=None,
                 path_to_model=None, alpha=1.0,
                 limit_batches=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_generator = DataGenerator(PATH_TO_TRAIN, self.batch_size, limit_batches=limit_batches,
                                             apply_mosaic=True)
        self.validation_generator = DataGenerator(PATH_TO_VALIDATION, self.batch_size, limit_batches=limit_batches)
        self.T = T if T is not None else 2 * epochs
        self.n_min = n_min
        self.n_max = n_max
        self.alpha = alpha

        self._model, self.true_boxes = build_model(alpha=self.alpha)
        if path_to_model is not None:
            self.load_model(path_to_model)

    @Trainer.model.getter
    def model(self):
        return self._model

    def load_model(self, path_to_model: str):
        self.model.load_weights(f"weights/{path_to_model}")

    def train(self, name: str):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-8,
                                                              decay=0.0
                                                              ),
                           loss=YoloLoss(anchors=self.train_generator.anchors, true_boxes=self.true_boxes))

        history = self.model.fit(self.train_generator, validation_data=self.validation_generator, epochs=self.epochs,
                                 callbacks=[CosineAnnealingScheduler(self.n_min, self.n_max, self.T),
                                            ModelCheckpoint(
                                                filepath=f"weights/{name}",
                                                save_best_only=True,
                                                verbose=2
                                            ),
                                            TerminateOnNaN(),
                                            EarlyStopping(patience=5, min_delta=1e-4, verbose=2),
                                            ReduceLROnPlateau(patience=3, factor=0.2),
                                            TensorBoard(log_dir=f"info_about_runs/{name}")
                                            ]
                                 , workers=os.cpu_count())
        return history


class ExtendedTrainer(Trainer):
    def __init__(self, trainer):
        self.trainer = trainer

    def train(self, name: str):
        return self.trainer.train(name)

    def load_model(self, path_to_model: str):
        self.trainer.load_model(path_to_model)

    @Trainer.model.getter
    def model(self):
        return self.trainer.model


class FineTuneTrainer(ExtendedTrainer):
    def __init__(self, trainer):
        super().__init__(trainer)

    def train(self, name: str):
        self.model.trainable = True
        return super().train(name)

    def load_model(self, path_to_model: str):
        self.model.trainable = True
        super().load_model(path_to_model)


class LogTrainer(ExtendedTrainer):
    def __init__(self, trainer):
        super().__init__(trainer)

    def train(self, name: str):
        self.model.summary()
        history = super().train(name)
        plot_history(history.history)
        return history


def compose_trainer(concrete_trainer, decorators):
    trainer = concrete_trainer
    for DecoratorClass in decorators:
        trainer = DecoratorClass(trainer)
    return trainer


def train():
    trainer = SimpleTrainer(epochs=50, n_min=1e-6, n_max=1e-3, T=60, path_to_model=None)
    trainer = compose_trainer(trainer, [LogTrainer])
    # trainer.train(name="model_v47.h5")
    trainer.train(name="coco_v3.h5")


def fine_tune():
    trainer = SimpleTrainer(epochs=50, n_min=1e-8, n_max=1e-5, T=60, path_to_model="model_v42.h5")
    trainer = compose_trainer(trainer, [LogTrainer, FineTuneTrainer])
    trainer.train(name="model_v42_fine_tuned.h5")


if __name__ == '__main__':
    # tf.keras.applications.mobilenet_v2.MobileNetV2().summary()
    # build_model()[0].summary()
    #train()
    fine_tune()
