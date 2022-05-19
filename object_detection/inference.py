import random
from typing import List

from tensorflow.keras import backend as K

from loss import create_cell_grid
from train import *


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


def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    x_stabilized = x - K.max(x, axis=-1, keepdims=True)

    return K.exp(x_stabilized / t) / K.sum(K.exp(x_stabilized / t), axis=-1, keepdims=True)


def output_processor(output, anchors, apply_argmax=True):
    cell_grid = create_cell_grid(len(anchors))
    cell_size = IMAGE_SIZE / GRID_SIZE
    xy = (K.sigmoid(output[..., :2]) + cell_grid) * cell_size
    wh = K.exp(output[..., 2:4]) * anchors
    conf_scores = K.sigmoid(output[..., 4:5])
    classes = conf_scores * softmax(output[..., 5:])

    if apply_argmax:
        conf_scores = K.max(classes, axis=-1)
        classes = K.argmax(classes)
    else:
        tf.squeeze(conf_scores)
    return conf_scores, \
           K.clip(tf.concat([xy - wh / 2, xy + wh / 2], axis=-1), min_value=0, max_value=IMAGE_SIZE), \
           classes


def non_max_suppression(y_pred, anchors, iou_threshold, score_threshold, batch_size=BATCH_SIZE):
    _, boxes, classes = output_processor(y_pred, anchors, apply_argmax=False)
    boxes = tf.reshape(boxes, (-1, GRID_SIZE * GRID_SIZE * len(anchors), 4))
    boxes = tf.expand_dims(boxes, axis=2)
    classes = tf.reshape(classes, (-1, GRID_SIZE * GRID_SIZE * len(anchors), len(ENCODE_LABEL)))
    nms_boxes, nms_scores, nms_classes, nms_valid = tf.image.combined_non_max_suppression(
        boxes, classes, MAX_BOXES_PER_IMAGES, MAX_BOXES_PER_IMAGES * batch_size, iou_threshold=iou_threshold,
        score_threshold=score_threshold, clip_boxes=False)
    return nms_scores, nms_boxes, nms_classes, nms_valid


def inference(model, inputs, score_threshold=0.6, iou_threshold=0.5, batch_size=BATCH_SIZE, use_predict_fn=True):
    anchors = process_anchors(ANCHORS_PATH)
    dummy_array = np.zeros((1, 1, 1, 1, MAX_BOXES_PER_IMAGES, 4))
    if use_predict_fn:
        y_pred = model.predict([inputs, dummy_array])
    else:
        y_pred = model([inputs, dummy_array])
    return non_max_suppression(y_pred, anchors, iou_threshold, score_threshold, batch_size)


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
            bboxes.append(bbox)
        output_boxes.append(bboxes)
    return output_boxes


def image_to_bounding_boxes(model, path, score_threshold=0.4, iou_threshold=0.3):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = np.expand_dims(image, axis=0)
    scores, boxes, classes, valid_detections = inference(model, image,
                                                         score_threshold=score_threshold,
                                                         iou_threshold=iou_threshold,
                                                         batch_size=1)
    scores, boxes, classes, valid_detections = K.get_value(scores), \
                                               K.get_value(boxes), \
                                               K.get_value(classes), \
                                               K.get_value(valid_detections)

    return extract_boxes(scores, boxes, classes, valid_detections)[0]


def draw_images(images, scores, boxes, classes, valid_detections):
    output_boxes = extract_boxes(scores, boxes, classes, valid_detections)
    return [with_bounding_boxes(img, bboxes) for img, bboxes in zip(images, output_boxes)]


def run_on_one_image(path, score_threshold):
    model, true_boxes = build_model()
    if "fine_tuned" in PATH_TO_MODEL:
        model.trainable = True
    model.load_weights(PATH_TO_MODEL)
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = np.expand_dims(image, axis=0)
    start = time.perf_counter()
    scores, boxes, classes, valid_detections = inference(model, image, score_threshold=score_threshold,
                                                         iou_threshold=0.3, batch_size=1)
    scores, boxes, classes, valid_detections = K.get_value(scores), \
                                               K.get_value(boxes), \
                                               K.get_value(classes), \
                                               K.get_value(valid_detections)

    print("time to run: ", time.perf_counter() - start)
    print("Shapes")
    print(f"Scores: {scores.shape}")
    print(f"Boxes: {boxes.shape}")
    print(f"Classes: {classes.shape}")
    print(f"Valid detections: {valid_detections.shape}")
    print()

    print(valid_detections)
    print(scores[0, 0:valid_detections[0]])
    pred_images_with_boxes = draw_images(image, scores, boxes, classes, valid_detections)
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Original")
    plt.imshow(image[0])

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Predicted")
    plt.imshow(pred_images_with_boxes[0])

    plt.tight_layout()
    plt.show()


def test(score_threshold=0.2, iou_threshold=0.3, batch_size=BATCH_SIZE, batch=None):
    generator = DataGenerator(PATH_TO_TEST, shuffle=False, batch_size=batch_size)
    print('#Batches:', len(generator))
    model, true_boxes = build_model()
    if "fine_tuned" in PATH_TO_MODEL:
        model.trainable = True
    model.load_weights(PATH_TO_MODEL)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=YoloLoss(anchors=generator.anchors, true_boxes=true_boxes, enable_logs=True))

    if batch is None:
        batch = random.randint(0, len(generator))
    print(f"batch {batch}")
    ground_truth, y_true = generator[batch]
    images, true_boxes = ground_truth[0], ground_truth[1]

    loss = model.evaluate(ground_truth, y_true, verbose=2)
    print(f"loss: {loss}")

    start = time.perf_counter()
    scores, boxes, classes, valid_detections = inference(model, images, score_threshold=score_threshold, iou_threshold=iou_threshold)
    scores, boxes, classes, valid_detections = K.get_value(scores), \
                                               K.get_value(boxes), \
                                               K.get_value(classes), \
                                               K.get_value(valid_detections)

    print("time to run: ", time.perf_counter() - start)
    print("Shapes")
    print(f"Scores: {scores.shape}")
    print(f"Boxes: {boxes.shape}")
    print(f"Classes: {classes.shape}")
    print(f"Valid detections: {valid_detections.shape}")

    pred_images_with_boxes = draw_images(images, scores, boxes, classes, valid_detections)
    true_images_with_boxes = draw_images(images, *process_ground_truth(y_true, len(generator.anchors)))
    size = y_true.shape[0]
    fig, axs = plt.subplots(size, 2, figsize=(20, 40))
    for i in range(0, size):
        axs[i][0].imshow(true_images_with_boxes[i])
        axs[i][1].imshow(pred_images_with_boxes[i])
        axs[i][0].axis("off")
        axs[i][1].axis("off")

    plt.tight_layout()
    #plt.savefig(f"documentation/results/final_results_model_v42/{batch}.jpeg")
    plt.show()


if __name__ == '__main__':
    PATH_TO_MODEL = "weights/model_v42_fine_tuned.h5"
    test(score_threshold=0.4, batch_size=8, batch=9)
    #run_on_one_image("documentation\\examples\\bus.jpg", 0.4)
    #run_on_one_image("documentation\\examples\\bus2.jpg", 0.37)
    #run_on_one_image("documentation\\examples\\bus3.jpg", 0.4)
    #run_on_one_image("documentation\\examples\\bus4.jpg", 0.4)

    #run_on_one_image("documentation\\examples\\busses.jpg", 0.4)
    #run_on_one_image("documentation\\examples\\busses2.png", 0.5)
    #run_on_one_image("documentation\\examples\\ubb_bus.jpg", 0.4)
