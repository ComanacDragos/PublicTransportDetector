import sys
import time

import numpy as np

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
                           no_classes=3, enable_logs=False):
    start = time.time()
    true_boxes_all = extract_boxes(*process_ground_truth(y_true, len(anchors)))
    scores, boxes, classes, valid_detections = non_max_suppression(y_pred, anchors,
                                                                   max_boxes, nms_iou_threshold, score_threshold,
                                                                   enable_logs)
    scores, boxes, classes, valid_detections = K.get_value(scores), \
                                               K.get_value(boxes), \
                                               K.get_value(classes), \
                                               K.get_value(valid_detections)

    pred_boxes_all = extract_boxes(scores, boxes, classes, valid_detections)
    if enable_logs:
        print(f"NMS time: {time.time() - start}")

    start = time.time()
    mAP_list = []
    ap_all_list = []
    for true_boxes, pred_boxes in zip(true_boxes_all, pred_boxes_all):
        class_to_box = {c: [] for c in range(no_classes)}
        total_positives = {c: 0 for c in range(no_classes)}
        for pred_box in pred_boxes:
            correct = False
            for true_box in true_boxes:
                if iou_bbox(pred_box, true_box) >= iou_threshold:
                    correct = True
                    total_positives[pred_box.c] += 1
                    break
            class_to_box[pred_box.c].append((correct, pred_box.score))

        ap_list = []

        for c in range(no_classes):
            #if len(class_to_box[c]) == 0:
            #    ap_list.append(1.0)
            #    continue
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
            ap_list.append(voc_ap(recall, precision))
        ap_all_list.append(ap_list)
        mAP_list.append(np.mean(ap_list))
    if enable_logs:
        print(f"Compute mAP time: {time.time()-start}")
    return np.mean(mAP_list), np.mean(np.asarray(ap_all_list), axis=0)


def evaluate_model(model: tf.keras.Model, generator: DataGenerator, iou_threshold, nms_iou_threshold,
                   score_threshold, max_boxes,
                   no_classes=3, enable_logs=False):
    mAPs = []
    average_precisions = []
    for i in range(len(generator)):
        data, y_true = generator[i]
        y_pred = model(data)
        start = time.time()
        mAP, aPs = mean_average_precision(y_true, y_pred, generator.anchors, iou_threshold, nms_iou_threshold,
                                     score_threshold, max_boxes, no_classes, enable_logs)

        if enable_logs:
            print(f"{i+1}/{len(generator)} - mAP: {mAP} - ap: {aPs} - time: {time.time()-start}")
        mAPs.append(mAP)
        average_precisions.append(aPs)
    return np.mean(mAPs), np.mean(np.asarray(average_precisions), axis=0)


if __name__ == '__main__':
    model, true_boxes = build_model()
    #model.trainable = True
    model.load_weights("weights/model_v4_2.h5")
    generator = DataGenerator(PATH_TO_TEST, batch_size=32, shuffle=False)

    mAP, aps = evaluate_model(model, generator, 0.5, 0.5, 0.3, MAX_BOXES_PER_IMAGES, enable_logs=True)
    print("mAP: ", mAP)
    for c in range(3):
        print(f"{DECODE_LABEL[c]} : {aps[c]}")
