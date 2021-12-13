import time

import tensorflow as tf
from generator import *
from train import build_simple_model
from generator import process_anchors
from loss import create_cell_grid
from tensorflow.keras import backend as K


def output_processor(output, anchors):
    cell_grid = create_cell_grid(len(anchors))
    cell_size = IMAGE_SIZE / GRID_SIZE
    xy = (K.sigmoid(output[..., :2]) + cell_grid) * cell_size
    wh = K.clip(K.exp(output[..., 2:4]) * anchors, min_value=0, max_value=IMAGE_SIZE)
    conf_scores = K.sigmoid(output[..., 4])
    classes = tf.expand_dims(conf_scores, -1) * K.softmax(output[..., 5:])
    return tf.concat([xy-wh//2, xy+wh//2], axis=-1), classes


def filter_boxes(boxes, classes, threshold, no_anchors):
    predicted_classes = K.argmax(classes)
    max_probabilities = K.max(classes, axis=-1)
    filtering_mask = K.greater_equal(max_probabilities, threshold)
    scores = tf.reshape(tf.boolean_mask(max_probabilities, filtering_mask), (-1, GRID_SIZE, GRID_SIZE, no_anchors))
    boxes_filtered = tf.reshape(tf.boolean_mask(boxes, filtering_mask), (-1, GRID_SIZE, GRID_SIZE, no_anchors, 4))
    classes_filtered = tf.reshape(tf.boolean_mask(predicted_classes, filtering_mask), (-1, GRID_SIZE, GRID_SIZE, no_anchors))

    return scores, boxes_filtered, classes_filtered


def non_max_suppression_for_one(scores, boxes, classes, max_boxes, iou_threshold):
    nms_indices = tf.image.non_max_suppression(boxes, scores,
                                               max_boxes, iou_threshold=iou_threshold)
    scores_filtered = K.gather(scores, nms_indices)
    boxes_filtered = K.gather(boxes, nms_indices)
    classes_filtered = K.gather(classes, nms_indices)

    return K.get_value(scores_filtered), K.get_value(boxes_filtered), K.get_value(classes_filtered)


def inference(model, inputs, score_threshold=0.6, iou_threshold=0.5, max_boxes=10, anchors_path=ANCHORS_PATH):
    anchors = process_anchors(anchors_path)
    dummy_array = np.zeros((1, 1, 1, 1, MAX_BOXES_PER_IMAGES, 4))
    start = time.time()
    y_pred = model.predict([inputs, dummy_array])
    print("predict time: ", time.time() - start)
    boxes, classes = output_processor(y_pred, anchors)
    scores, boxes, classes = filter_boxes(boxes, classes, score_threshold, len(anchors))
    scores, boxes, classes = K.get_value(scores), K.get_value(boxes), K.get_value(classes)
    print(scores.shape, boxes.shape, classes.shape)
    for i in range(len(inputs)):
        scores_i, boxes_i, classes_i = non_max_suppression_for_one(tf.keras.backend.flatten(scores[i]),
                                                                   tf.reshape(boxes[i], (-1, 4)),
                                                                   tf.keras.backend.flatten(classes[i]),
                                                                   max_boxes, iou_threshold)
        print(scores_i.shape, boxes_i.shape, classes_i.shape)

    return K.get_value(scores), K.get_value(boxes), K.get_value(classes)


def test():
    generator = DataGenerator(PATH_TO_TRAIN, 8)
    model, _ = build_simple_model()
    model.load_weights("weights/model.h5")
    # model = tf.keras.models.load_model("weights/model.h5", compile=False)
    ground_truth, y_true = generator[0]
    images, true_boxes = ground_truth[0], ground_truth[1]

    start = time.time()
    scores, boxes, classes = inference(model, images, score_threshold=0.6, iou_threshold=0.5)
    print("time to run: ", time.time() - start)
    #print(scores.shape, boxes.shape, classes.shape)
    #print(scores)
    #print(boxes)
    #print(classes)
    return
    fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 2]})
    for i in range(0, 1):
        original_boxes = interpret_output(y_true[i - 1], generator.anchors)
        predicted_boxes = interpret_output(y_pred[i - 1], generator.anchors)

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


if __name__ == '__main__':
    test()
