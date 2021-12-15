import time

import numpy as np
import tensorflow as tf
from generator import *
from train import build_mobilenet_model
from generator import process_anchors
from loss import create_cell_grid, YoloLoss
from tensorflow.keras import backend as K
from typing import List


def process_ground_truth_for_one(ground_truth, no_anchors):
    conf_scores, boxes, classes = [], [], []
    cell_size = IMAGE_SIZE / GRID_SIZE
    for cx in range(GRID_SIZE):
        for cy in range(GRID_SIZE):
            for a in range(no_anchors):
                anchor = ground_truth[cx][cy][a]
                if anchor[4] == 1:
                    conf_scores.append(anchor[4])
                    x, y, w, h = anchor[:4]
                    x = x * cell_size
                    y = y * cell_size
                    boxes.append([x - w // 2, y - h // 2, x + w // 2, y + h // 2])
                    classes.append(np.argmax(anchor[5:], axis=-1))

    return np.asarray(conf_scores), np.asarray(boxes), np.asarray(classes)


def process_ground_truth(ground_truth, no_anchors):
    conf_scores, boxes, classes = [], [], []
    for i in range(ground_truth.shape[0]):
        conf_scores_i, boxes_i, classes_i = process_ground_truth_for_one(ground_truth[i], no_anchors)
        conf_scores.append(conf_scores_i)
        boxes.append(boxes_i)
        classes.append(classes_i)
    return np.asarray(conf_scores), np.asarray(boxes), np.asarray(classes)


def output_processor(output, anchors):
    cell_grid = create_cell_grid(len(anchors))
    cell_size = IMAGE_SIZE / GRID_SIZE
    xy = (K.sigmoid(output[..., :2]) + cell_grid) * cell_size
    wh = K.exp(output[..., 2:4]) * anchors
    conf_scores = K.sigmoid(output[..., 4])
    classes = tf.expand_dims(conf_scores, -1) * K.softmax(output[..., 5:])
    return conf_scores, \
           K.clip(tf.concat([xy - wh // 2, xy + wh // 2], axis=-1), min_value=0, max_value=IMAGE_SIZE), \
           K.argmax(classes)


def non_max_suppression_for_one_aux(scores, boxes, classes, max_boxes, iou_threshold, score_threshold):
    nms_indices = tf.image.non_max_suppression(boxes, scores,
                                               max_boxes, iou_threshold=iou_threshold, score_threshold=score_threshold)
    scores_filtered = K.gather(scores, nms_indices)
    boxes_filtered = K.gather(boxes, nms_indices)
    classes_filtered = K.gather(classes, nms_indices)

    return scores_filtered, boxes_filtered, classes_filtered


def non_max_suppression_for_one(scores, boxes, classes, max_boxes, iou_threshold, score_threshold):
    return non_max_suppression_for_one_aux(tf.keras.backend.flatten(scores),
                                           tf.reshape(boxes, (-1, 4)),
                                           tf.keras.backend.flatten(classes),
                                           max_boxes, iou_threshold, score_threshold)


def non_max_suppression(y_pred, anchors, max_boxes, iou_threshold, score_threshold, enable_logs=False):
    scores, boxes, classes = output_processor(y_pred, anchors)

    if enable_logs:
        print(K.get_value(tf.keras.backend.min(y_pred[..., 4])), K.get_value(tf.keras.backend.max(y_pred[..., 4])))
        print(K.get_value(tf.keras.backend.min(y_pred[..., 2])), K.get_value(tf.keras.backend.max(y_pred[..., 2])))
        print(K.get_value(tf.keras.backend.min(y_pred[..., 3])), K.get_value(tf.keras.backend.max(y_pred[..., 3])))

    scores, boxes, classes = K.get_value(scores), K.get_value(boxes), K.get_value(classes)

    if enable_logs:
        print(scores.shape, boxes.shape, classes.shape)
        print(K.get_value(tf.keras.backend.min(scores)), K.get_value(tf.keras.backend.max(scores)))
        y, _ = K.get_value(tf.unique(K.get_value(K.flatten(y_pred[..., 4]))))
        print(K.get_value(y))
        y, _ = K.get_value(tf.unique(K.get_value(K.flatten(scores))))
        print(K.get_value(y))

    output_scores, output_boxes, output_classes = [], [], []
    for i in range(len(boxes)):
        start = time.time()
        scores_i, boxes_i, classes_i = non_max_suppression_for_one(scores[i], boxes[i], classes[i],
                                                                   max_boxes, iou_threshold, score_threshold)
        output_scores.append(K.get_value(scores_i))
        output_boxes.append(K.get_value(boxes_i))
        output_classes.append(K.get_value(classes_i))

        if enable_logs:
            print(i, "time: ", time.time() - start)
            print(output_scores[-1].shape)
            print(output_boxes[-1].shape)
            print(output_classes[-1].shape)
    """output_scores, output_boxes, output_classes = tf.map_fn(lambda x: non_max_suppression_for_one(x[0], x[1], x[2],
                                                                                                  max_boxes,
                                                                                                  iou_threshold,
                                                                                                  score_threshold),
                                                            (scores, boxes, classes)
                                                            )
    """
    return output_scores, output_boxes, output_classes


def inference(model, inputs, score_threshold=0.6, iou_threshold=0.5, max_boxes=MAX_BOXES_PER_IMAGES,
              anchors_path=ANCHORS_PATH, enable_logs=False):
    anchors = process_anchors(anchors_path)
    dummy_array = np.zeros((1, 1, 1, 1, MAX_BOXES_PER_IMAGES, 4))
    # start = time.time()
    y_pred = model([inputs, dummy_array], training=True)
    # print("predict time: ", time.time() - start)
    return non_max_suppression(y_pred, anchors, max_boxes, iou_threshold, score_threshold, enable_logs)


def extract_boxes(scores, boxes, classes) -> List[List[BoundingBox]]:
    output_boxes = []
    for i in range(len(boxes)):
        bboxes = []
        for j in range(len(boxes[i])):
            bbox = BoundingBox(classes[i][j],
                               boxes[i][j][0],
                               boxes[i][j][1],
                               boxes[i][j][2],
                               boxes[i][j][3],
                               scores[i][j]
                               )
            w, h = bbox.width_height()
            x, y = bbox.center()
            cell_size = IMAGE_SIZE / GRID_SIZE
            x = int(x / cell_size)
            y = int(y / cell_size)
            if w > 50 and h > 50:
                bboxes.append(bbox)
        output_boxes.append(bboxes)
    return output_boxes


def draw_images(images, scores, boxes, classes):
    output_boxes = extract_boxes(scores, boxes, classes)
    return [with_bounding_boxes(img, bboxes) for img, bboxes in zip(images, output_boxes)]


def test():
    generator = DataGenerator(PATH_TO_TRAIN, 8, shuffle=False)
    model, true_boxes = build_mobilenet_model()
    model.load_weights("weights/model.h5")
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=YoloLoss(anchors=generator.anchors, true_boxes=true_boxes, enable_logs=True))

    ground_truth, y_true = generator[0]
    images, true_boxes = ground_truth[0], ground_truth[1]

    loss = model.evaluate(ground_truth, y_true, verbose=2)
    print(f"loss: {loss}")

    start = time.time()
    scores, boxes, classes = inference(model, images, score_threshold=0.0, iou_threshold=0.1,
                                       max_boxes=MAX_BOXES_PER_IMAGES, enable_logs=True)
    scores, boxes, classes = K.get_value(scores), K.get_value(boxes), K.get_value(classes)

    print("time to run: ", time.time() - start)
    pred_images_with_boxes = draw_images(images, scores, boxes, classes)
    true_images_with_boxes = draw_images(images, *process_ground_truth(y_true, len(generator.anchors)))
    fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 1]})
    for i in range(0, 3):
        # plt.subplot(6, 2, 2 * i - 1)
        axs[i][0].imshow(true_images_with_boxes[i])
        # axs[i][0].title('Original')

        # plt.subplot(6, 2, 2 * i)
        axs[i][1].imshow(pred_images_with_boxes[i])
        # axs[i][1].title('Predicted')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()
