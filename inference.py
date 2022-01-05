import random
import sys
import time

import numpy as np
import tensorflow as tf
from generator import *
from train import *
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
    conf_scores, boxes, classes, valid_detections = [], [], [], []
    for i in range(ground_truth.shape[0]):
        conf_scores_i, boxes_i, classes_i = process_ground_truth_for_one(ground_truth[i], no_anchors)
        conf_scores.append(conf_scores_i)
        boxes.append(boxes_i)
        classes.append(classes_i)
        valid_detections.append(len(boxes_i))
    return np.asarray(conf_scores), np.asarray(boxes), np.asarray(classes), np.asarray(valid_detections)


def output_processor(output, anchors, apply_argmax=True):
    cell_grid = create_cell_grid(len(anchors))
    cell_size = IMAGE_SIZE / GRID_SIZE
    xy = (K.sigmoid(output[..., :2]) + cell_grid) * cell_size
    wh = K.exp(output[..., 2:4]) * anchors
    conf_scores = K.sigmoid(output[..., 4])
    classes = tf.expand_dims(conf_scores, -1) * K.softmax(output[..., 5:])
    if apply_argmax:
        classes = K.argmax(classes)
    return conf_scores, \
           K.clip(tf.concat([xy - wh // 2, xy + wh // 2], axis=-1), min_value=0, max_value=IMAGE_SIZE), \
           classes


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


def non_max_suppression_slow(y_pred, anchors, max_boxes, iou_threshold, score_threshold, enable_logs=False):
    start = time.time()
    scores, boxes, classes = output_processor(y_pred, anchors)

    if enable_logs:
        print(K.get_value(tf.keras.backend.min(y_pred[..., 4])), K.get_value(tf.keras.backend.max(y_pred[..., 4])))
        print(K.get_value(tf.keras.backend.min(y_pred[..., 2])), K.get_value(tf.keras.backend.max(y_pred[..., 2])))
        print(K.get_value(tf.keras.backend.min(y_pred[..., 3])), K.get_value(tf.keras.backend.max(y_pred[..., 3])))

    scores, boxes, classes = K.get_value(scores), K.get_value(boxes), K.get_value(classes)
    print(f"Process time: {time.time() - start}")
    if enable_logs:
        print(scores.shape, boxes.shape, classes.shape)
        print(K.get_value(tf.keras.backend.min(scores)), K.get_value(tf.keras.backend.max(scores)))
        y, _ = K.get_value(tf.unique(K.get_value(K.flatten(y_pred[..., 4]))))
        print("unique raw conf scores: ", K.get_value(y))
        y, _ = K.get_value(tf.unique(K.get_value(K.flatten(scores))))
        print(K.get_value(y))
        print(K.get_value(K.sigmoid(y)))


    output_scores, output_boxes, output_classes = [], [], []
    start_for = time.time()
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
    print(f"For time: {time.time() - start_for}")
    return output_scores, output_boxes, output_classes


def non_max_suppression_fast(y_pred, anchors, max_boxes, iou_threshold, score_threshold, enable_logs=False):
    # boxes : bs, 13, 13, 3, 4
    _, boxes, classes = output_processor(y_pred, anchors, apply_argmax=False)
    boxes = tf.reshape(boxes, (-1, GRID_SIZE*GRID_SIZE*len(anchors), 4))
    boxes = tf.expand_dims(boxes, axis=2)
    classes = tf.reshape(classes, (-1, GRID_SIZE*GRID_SIZE*len(anchors), len(ENCODE_LABEL)))
    nms_boxes, nms_scores, nms_classes, nms_valid = tf.image.combined_non_max_suppression(
        boxes, classes, max_boxes, max_boxes * 8, iou_threshold=iou_threshold,
        score_threshold=score_threshold, clip_boxes=False)
    return nms_scores, nms_boxes, nms_classes, nms_valid


def non_max_suppression(y_pred, anchors, max_boxes, iou_threshold, score_threshold, enable_logs=False):
    return non_max_suppression_fast(y_pred, anchors, max_boxes, iou_threshold, score_threshold, enable_logs=enable_logs)


def inference(model, inputs, score_threshold=0.6, iou_threshold=0.5, max_boxes=MAX_BOXES_PER_IMAGES,
              anchors_path=ANCHORS_PATH, enable_logs=False):
    anchors = process_anchors(anchors_path)
    dummy_array = np.zeros((1, 1, 1, 1, MAX_BOXES_PER_IMAGES, 4))
    # start = time.time()
    y_pred = model([inputs, dummy_array])
    y, _ = K.get_value(tf.unique((K.flatten(y_pred[..., 4]))))
    print("unique raw conf scores: ", K.get_value(y))
    print(K.get_value(K.sigmoid(y)))
    print("len ", K.get_value(tf.shape(y)))
    print("max obj score ", K.get_value(K.max(K.sigmoid(y))))
    _, boxes, classes = output_processor(y_pred, anchors, apply_argmax=False)
    print("max class score ", K.get_value(K.max(K.reshape(classes, (BATCH_SIZE, -1)), axis=-1)))


    ##for i in range(13):
    #    for j in range(13):
    #        for a in range(3):
    #            print(K.get_value(y_pred[0, i,j, a, :]))
    # print("predict time: ", time.time() - start)
    return non_max_suppression(y_pred, anchors, max_boxes, iou_threshold, score_threshold, enable_logs)


def extract_boxes(scores, boxes, classes, valid_detections) -> List[List[BoundingBox]]:
    output_boxes = []
    for i in range(len(boxes)):
        bboxes = []
        for j in range(valid_detections[i]):
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
            #if w > 50 and h > 50:
            bboxes.append(bbox)
            #print(bbox.as_coordinates_array(), bbox.c, bbox.score)
        output_boxes.append(bboxes)
    return output_boxes


def draw_images(images, scores, boxes, classes, valid_detections):
    output_boxes = extract_boxes(scores, boxes, classes, valid_detections)
    return [with_bounding_boxes(img, bboxes) for img, bboxes in zip(images, output_boxes)]


def test():
    generator = DataGenerator(PATH_TO_TEST, shuffle=False)
    model, true_boxes = build_model()
    #model.trainable = True
    model.load_weights("weights/model_v5.h5")

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=YoloLoss(anchors=generator.anchors, true_boxes=true_boxes, enable_logs=True))

    batch = random.randint(0, len(generator))
    print(f"batch {batch}")
    ground_truth, y_true = generator[batch]
    images, true_boxes = ground_truth[0], ground_truth[1]

    loss = model.evaluate(ground_truth, y_true, verbose=2)
    print(f"loss: {loss}")

    start = time.time()
    scores, boxes, classes, valid_detections = inference(model, images, score_threshold=0.20, iou_threshold=0.3,
                                       max_boxes=MAX_BOXES_PER_IMAGES, enable_logs=True)
    scores, boxes, classes, valid_detections = K.get_value(scores),\
                                               K.get_value(boxes),\
                                               K.get_value(classes),\
                                               K.get_value(valid_detections)

    print("time to run: ", time.time() - start)

    print(scores[0], "scores")
    print(boxes[0], "boxes")
    print(classes[0], "classes")
    print(valid_detections[0], "detections")
    print("===============")
    pred_images_with_boxes = draw_images(images, scores, boxes, classes, valid_detections)
    true_images_with_boxes = draw_images(images, *process_ground_truth(y_true, len(generator.anchors)))
    fig, axs = plt.subplots(8, 2, figsize=(30, 30))
    for i in range(0, 8):
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
    """boxes = np.asarray([[[3, 2, 4, 4], [2, 2, 4, 4]]])
    boxes = np.expand_dims(boxes, 2)
    print(boxes.shape)
    classes = np.asarray([[[0.2, 0.6, 0.2], [0.2, 0.7, 0.1]]])
    print(classes.shape)
    scores, boxes, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes, classes, 2, 4, iou_threshold=0.5,
        score_threshold=0.5, clip_boxes=False)
    boxes, scores, classes, valid_detections = K.get_value(scores), \
                                               K.get_value(boxes), \
                                               K.get_value(classes), \
                                               K.get_value(valid_detections)
    print(boxes)
    print(scores)
    print(classes)
    print(valid_detections)"""