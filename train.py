import time

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from generator import *


def conv_layer(x, kernel_size=3, filters=32, reluActivation=True, strides=1):
    conv = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, padding="same", activation="relu",
                                  strides=strides,
                                  kernel_initializer=tf.keras.initializers.HeNormal())(x)
    bn = tf.keras.layers.BatchNormalization()(conv)
    if reluActivation:
        return tf.keras.layers.ReLU()(bn)
    return bn


def build_simple_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    mobilenet_v2.trainable = False
    x = mobilenet_v2(x, training=False)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(x, filters=320)
    x = conv_layer(x, filters=320)
    x = conv_layer(x, filters=160)
    x = conv_layer(x, filters=160)
    output = conv_layer(x, filters=40, reluActivation=False)
    return tf.keras.Model(inputs=inputs, outputs=output, name="custom_yolo")


def upsample_block(x, filters, size, stride=2):
    """
    x - the input of the upsample block
    filters - the number of filters to be applied
    size - the size of the filters
    """
    x = tf.keras.layers.Convolution2DTranspose(kernel_size=size, filters=filters, strides=stride, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), trainable=False):
    # define the input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    downsample_skip_layer_name = ["block_6_expand_relu",
                                  "block_10_expand_relu",
                                  "block_14_expand_relu"]

    down_stack = tf.keras.Model(inputs=mobilenet_v2.input,
                                outputs=[mobilenet_v2.get_layer(name).output for name in downsample_skip_layer_name],
                                name="down_stack")
    down_stack.trainable = trainable

    skips = down_stack(x, training=trainable)
    x = skips[-1]

    for skip_layer in reversed(skips[:-1]):
        x = upsample_block(x, skip_layer.shape[-1], 3)
        x = tf.keras.layers.Concatenate()([x, skip_layer])

    x = conv_layer(x, filters=192, strides=2)
    x = conv_layer(x, filters=40, strides=2, reluActivation=False)
    output = x
    return tf.keras.Model(inputs=inputs, outputs=output)


def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), trainable=False, type="simple"):
    # if type == "simple":
    return build_simple_model(input_shape)
    # return build_unet(input_shape, trainable)


def process_prediction(prediction):
    # print(prediction[0, 0, 0, :][[4, 12, 20, 28, 36]])
    # print(prediction[0, 0, 1, :][[4, 12, 20, 28, 36]])

    x = tf.reshape(prediction, (-1, 13, 13, 5, 8))
    x = tf.cast(x, tf.float32)
    x = tf.where([True, True, False, False, True, False, False, False], tf.sigmoid(x), x)
    return tf.cast(x, tf.float32)


def plot_history(history):
    plt.plot(history['loss'], label='loss')
    # plt.plot(history['val_loss'], label='val_loss')
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


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, anchors, l_coord=1, l_noobj=0.1, enable_logs=False):
        super(YoloLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.anchors_width = anchors[:, 0]
        self.anchors_height = anchors[:, 1]
        self.x_offset = np.zeros((GRID_SIZE, GRID_SIZE))
        self.y_offset = np.zeros((GRID_SIZE, GRID_SIZE))
        self.cell_size = IMAGE_SIZE / GRID_SIZE
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.x_offset[i][j] = i * self.cell_size
                self.y_offset[i][j] = j * self.cell_size
        self.x_offset = np.stack([self.x_offset] * anchors.shape[0], axis=-1)
        self.y_offset = np.stack([self.y_offset] * anchors.shape[0], axis=-1)
        self.enable_logs = enable_logs

    def call(self, y_true: tf.Tensor, y_pred_raw: tf.Tensor):
        """
        Each anchor is composed of 8 values:
        0, 1: x, y position
        2, 3: width, height
        4: if there is an object
        5, 6, 7: probabilities

        y_true, y_pred : shape -> (batch_size, grid_size, grid_size, anchors, 8)
        """
        y_pred = tf.reshape(y_pred_raw, (-1, GRID_SIZE, GRID_SIZE, len(self.anchors_width), 8))

        conf_true, conf_pred = y_true[..., 4], K.sigmoid(y_pred[..., 4])
        x, x_hat = y_true[..., 0] * self.cell_size + self.x_offset, K.sigmoid(y_pred[..., 0]) * self.cell_size + self.x_offset
        y, y_hat = y_true[..., 1] * self.cell_size + self.y_offset, K.sigmoid(y_pred[..., 1]) * self.cell_size + self.y_offset
        w, w_hat = K.exp(y_true[..., 2]) * self.anchors_width, K.exp(y_pred[..., 2]) * self.anchors_width
        h, h_hat = K.exp(y_true[..., 3]) * self.anchors_height, K.exp(y_pred[..., 3]) * self.anchors_height
        class_scores, class_scores_hat = y_true[..., 5:], y_pred[..., 5:]

        axes_to_reduce = [1, 2, 3]
        no_true_boxes = sum_over_axes(conf_true, axes_to_reduce)

        xywh_loss = (self.l_coord / (no_true_boxes + 1e-9)) * \
                    (sum_over_axes(conf_true * ((x - x_hat) ** 2 + (y - y_hat) ** 2), axes_to_reduce) +
                     sum_over_axes(conf_true * ((w - w_hat) ** 2 + (h - h_hat) ** 2), axes_to_reduce))

        class_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=K.argmax(class_scores),
                                                                     y_pred=class_scores_hat,
                                                                     from_logits=True)
        class_loss = sum_over_axes(conf_true * class_loss, axes_to_reduce) / (no_true_boxes + 1e-9)

        truth_area = w * h
        predicted_area = w_hat * h_hat

        x_mins, x_maxes = x - w/2., x + w/2.
        y_mins, y_maxes = y - h/2., y + h/2.

        x_mins_hat, x_maxes_hat = x_hat - w_hat/2., x_hat + w_hat/2.
        y_mins_hat, y_maxes_hat = y_hat - h_hat/2., y_hat + h_hat/2.

        x_mins_i, y_mins_i = tf.minimum(x_mins, x_mins_hat), tf.minimum(y_mins, y_mins_hat)
        x_maxes_i, y_maxes_i = tf.minimum(x_maxes, x_maxes_hat), tf.minimum(y_maxes, y_maxes_hat)

        w_i, h_i = tf.maximum(x_maxes_i - x_mins_i, 0.), tf.maximum(y_maxes_i - y_mins_i, 0.)
        intersect_area = w_i * h_i
        union_area = predicted_area + truth_area - intersect_area
        iou = tf.truediv(intersect_area, union_area)

        print(iou.shape)

        """obj_diff = (objectness_scores * iou - objectness_scores_hat) ** 2

        obj_loss = tf.reduce_sum(objectness_scores * obj_diff) + \
                   self.l_noobj * tf.reduce_sum((1 - objectness_scores) * obj_diff)

        class_loss = tf.reduce_sum(
            tf.stack([objectness_scores] * 3, axis=-1) * ((class_scores - class_scores_hat) ** 2))

        # print(np.min(x), np.max(x))
        # print(np.min(y), np.max(y))
        # print(np.min(x_hat), np.max(x_hat))
        # print(np.min(y_hat), np.max(y_hat))

        if self.enable_logs:
            print(f"xy : {xy_loss}")
            print(f"wh : {wh_loss}")
            print(f"obj : {obj_loss}")
            print(f"class : {class_loss}")
            print()
        # tf.print(xy_loss + wh_loss + obj_loss + class_loss)
        """
        return 0  # xy_loss + wh_loss + obj_loss + class_loss


class Train:
    def __init__(self, epochs=5, batch_size=32, n_min=1e-5, n_max=4e-4, T=None, path_to_model=None, limit_batches=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_generator = DataGenerator(PATH_TO_TRAIN, self.batch_size, limit_batches=limit_batches)
        self.validation_generator = DataGenerator(PATH_TO_VALIDATION, self.batch_size, limit_batches=limit_batches)
        self.T = T if T is not None else 2 * batch_size
        self.n_min = n_min
        self.n_max = n_max
        self.model = None
        if path_to_model is None:
            self.new_model()
        else:
            self.load_model(path_to_model)

    def train(self, name="model.h5", fine_tune=False):
        self.model.summary()

        if fine_tune:
            self.model.trainable = True
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=YoloLoss(anchors=self.train_generator.anchors),
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
                                            # tf.keras.callbacks.EarlyStopping(patience=3, min_delta=1e-3, verbose=2)
                                            ], workers=os.cpu_count())
        tf.keras.models.save_model(self.model, f"weights/{name}")
        plot_history(history.history)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={
            "YoloLoss": YoloLoss
        }, compile=False)

    def new_model(self):
        self.model = build_model()


def visualize_predictions(path):
    test_generator = DataGenerator(PATH_TO_TEST, 16, shuffle=False)
    images, y_true = test_generator[1]
    model = tf.keras.models.load_model(path, compile=False, custom_objects={
        "YoloLoss": YoloLoss
    })
    y_pred = process_prediction(model.predict(images))

    plt.figure(figsize=(8, 7), dpi=160)
    # fig, axes = plt.subplots(nrows=6, ncols=2)

    for i in range(1, 7):
        original_boxes = interpret_output(y_true[i], test_generator.anchors)
        predicted_boxes = interpret_output(y_pred[i], test_generator.anchors)

        print(f"{i} th image:")
        for j in original_boxes:
            print(j)
        print("Predicted:")
        for j in predicted_boxes:
            print(j)

        print()
        original_image = with_bounding_boxes(images[i], original_boxes, 3, [200, 0, 0])
        predicted_image = with_bounding_boxes(images[i], predicted_boxes, 3, [200, 0, 0])

        plt.subplot(6, 2, 2 * i - 1)
        plt.imshow(original_image)
        plt.title('Original')

        plt.subplot(6, 2, 2 * i)
        plt.imshow(predicted_image)
        plt.title('Predicted')

    plt.tight_layout()
    plt.show()


def train():
    t = Train(epochs=5, batch_size=8, n_min=1e-7, n_max=4e-4, limit_batches=None)
    t.train(name="model")
    fine_tune = Train(epochs=3, batch_size=8, n_min=1e-7, n_max=1e-6, path_to_model="weights/model")
    fine_tune.train(name="model_fine_tuned", fine_tune=True)


# 0000599864fd15b3.jpg
# 0000599864fd15b3.jpg
def test():
    t = Train(epochs=20, batch_size=8, n_min=1e-7, n_max=1e-3, limit_batches=1)
    """
    model = build_simple_model()
    model.load_weights("weights/only_weights/model")
    
    """
    # model = tf.keras.models.load_model("weights/model", compile=False)

    images, y_true = t.train_generator[0]
    # y_pred = model(images)
    print(YoloLoss(t.train_generator.anchors, l_coord=1, enable_logs=True).call(y_true, y_true))
    # y_pred = process_prediction(y_pred)
    # fig, axes = plt.subplots(nrows=6, ncols=2)
    return
    fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 2]})
    for i in range(0, 1):
        original_boxes = interpret_output(y_true[i - 1], t.train_generator.anchors)
        predicted_boxes = interpret_output(y_pred[i - 1], t.train_generator.anchors)

        print(f"{i} th image:")
        for j in original_boxes:
            print(j)
        print("Predicted:")
        for j in predicted_boxes:
            print(j)

        print()
        original_image = with_bounding_boxes(images[i - 1], original_boxes, 3)
        predicted_image = with_bounding_boxes(images[i - 1], predicted_boxes, 3)

        # plt.subplot(6, 2, 2 * i - 1)
        axs[i][0].imshow(original_image)
        # axs[i][0].title('Original')

        # plt.subplot(6, 2, 2 * i)
        axs[i][1].imshow(predicted_image)
        # axs[i][1].title('Predicted')

    plt.tight_layout()
    plt.show()


# 146.3025
if __name__ == '__main__':
    # visualize_predictions("weights/model.h5")
    # Train(epochs=1).train()
    # model = build_unet()
    # model.summary()
    # test_loss()
    # train()
    test()
