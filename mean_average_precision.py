import gc

from inference import *


def voc_ap(rec, prec):
    """
    Official development kit matlab code VOC2012---
    function ap = VOCap(rec,prec)

    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    indexes = []  # indexes where the recall changes
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            indexes.append(i)

    ap = 0.0
    for i in indexes:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap


def mean_average_precision(y_true, y_pred, anchors, iou_threshold, nms_iou_threshold, score_threshold, max_boxes,
                           no_classes=3):
    start = time.perf_counter()
    true_boxes_all = extract_boxes(*process_ground_truth(y_true, len(anchors)))
    logging.info(f"Ground truth processing: {time.perf_counter() - start}")
    start = time.perf_counter()
    scores, boxes, classes, valid_detections = non_max_suppression(y_pred, anchors,
                                                                   max_boxes, nms_iou_threshold, score_threshold)
    scores, boxes, classes, valid_detections = K.get_value(scores), \
                                               K.get_value(boxes), \
                                               K.get_value(classes), \
                                               K.get_value(valid_detections)

    pred_boxes_all = extract_boxes(scores, boxes, classes, valid_detections)
    logging.info(f"NMS time: {time.perf_counter() - start}")

    start = time.perf_counter()
    ap_to_class = {c: [] for c in range(no_classes)}
    for true_boxes, pred_boxes in zip(true_boxes_all, pred_boxes_all):
        class_to_box = {c: [] for c in range(no_classes)}
        total_positives = {c: 0 for c in range(no_classes)}
        contains_class = {c: False for c in range(no_classes)}
        for pred_box in pred_boxes:
            correct = False
            for true_box in true_boxes:
                contains_class[true_box.c] = True
                if iou_bbox(pred_box, true_box) >= iou_threshold:
                    correct = True
                    total_positives[pred_box.c] += 1
                    break
            class_to_box[pred_box.c].append((correct, pred_box.score))

        for c in range(no_classes):
            if len(class_to_box[c]) == 0:
                if contains_class[c]:
                    ap_to_class[c].append(0.)
                continue

            sorted_boxes = class_to_box[c]
            sorted_boxes.sort(key=lambda x: x[1], reverse=True)
            recall = []
            precision = []
            running_corrects = 0
            for i, (correct, _) in enumerate(sorted_boxes, start=1):
                if correct:
                    running_corrects += 1
                precision.append(running_corrects / i)
                if total_positives[c] == 0:
                    recall.append(0)
                else:
                    recall.append(running_corrects / total_positives[c])
            ap_to_class[c].append(voc_ap(recall, precision))

    logging.info(f"Compute mAP time: {time.perf_counter() - start}")
    return ap_to_class


def evaluate_model(model: tf.keras.Model, generator: DataGenerator, iou_true_positive_threshold, nms_iou_threshold,
                   score_threshold, max_boxes,
                   no_classes=3):
    average_precisions = []
    ap_to_class_all = {c: [] for c in range(no_classes)}
    for i in range(len(generator)):
        start_total = time.perf_counter()
        start = time.perf_counter()
        data, y_true = generator[i]
        logging.info(f"Load data time: {time.perf_counter() - start}")
        y_pred = model.predict(data)
        start = time.perf_counter()

        ap_to_class = mean_average_precision(y_true, y_pred, generator.anchors, iou_true_positive_threshold,
                                             nms_iou_threshold,
                                             score_threshold, max_boxes, no_classes)

        logging.info(f"time mAP: {time.perf_counter() - start}")
        logging.info(f"{i + 1}/{len(generator)} time total: {time.perf_counter() - start_total}")
        for c, aps in ap_to_class.items():
            ap_to_class_all[c] += aps

    no_items = []
    for c, aps in ap_to_class_all.items():
        if len(aps) > 0:
            average_precisions.append(np.mean(aps))
        else:
            average_precisions.append(0.)
        no_items.append(len(aps))

    return np.mean(average_precisions), np.asarray(average_precisions), no_items


if __name__ == '__main__':
    model, true_boxes = build_model()
    #model.trainable = True
    model.load_weights("weights/model_v22.h5")
    generator = DataGenerator(PATH_TO_TEST, shuffle=False)

    mAP, aps, no_items = evaluate_model(model, generator, iou_true_positive_threshold=0.5,
                                                          nms_iou_threshold=0.3,
                                                          score_threshold=0.2,
                                                          max_boxes=MAX_BOXES_PER_IMAGES)
    print("mAP: ", mAP)
    print(f"Number of items: {no_items}")
    for c in range(3):
        print(f"{DECODE_LABEL[c]} : {aps[c]}")
