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


def mean_average_precision(y_true, y_pred, anchors, iou_threshold, nms_iou_threshold, score_threshold, no_classes=len(ENCODE_LABEL)):
    """
    Computes for each class the average precision for a batch

    :param y_true: ground truth
    :param y_pred: predicted
    :param anchors: list of anchors
    :param iou_threshold: minimum IOU for a true positive
    :param nms_iou_threshold: maximum IOU for NMS
    :param score_threshold: minimum score a bounding box can have
    :param no_classes: number of classes
    :return: dictionary mapping a class to it's corresponding average precision
    """
    start = time.perf_counter()
    true_boxes_all = extract_boxes(*process_ground_truth(y_true, len(anchors)))
    logging.debug(f"Ground truth processing: {time.perf_counter() - start}")
    start = time.perf_counter()
    scores, boxes, classes, valid_detections = non_max_suppression(y_pred, anchors, nms_iou_threshold, score_threshold)
    scores, boxes, classes, valid_detections = K.get_value(scores), \
                                               K.get_value(boxes), \
                                               K.get_value(classes), \
                                               K.get_value(valid_detections)

    pred_boxes_all = extract_boxes(scores, boxes, classes, valid_detections)
    logging.debug(f"NMS time: {time.perf_counter() - start}")

    start = time.perf_counter()
    ap_to_class = {c: [] for c in range(no_classes)}
    for true_boxes, pred_boxes in zip(true_boxes_all, pred_boxes_all):
        class_to_box = {c: [] for c in range(no_classes)}
        total_true_positives = {c: 0 for c in range(no_classes)}
        contains_class = {c: False for c in range(no_classes)}
        if len(pred_boxes) == 0:
            for true_box in true_boxes:
                contains_class[true_box.c] = True

        for pred_box in pred_boxes:
            correct = False
            for true_box in true_boxes:
                contains_class[true_box.c] = True
                if true_box.c == pred_box.c and iou_bbox(pred_box, true_box) >= iou_threshold:
                    correct = True
                    total_true_positives[pred_box.c] += 1
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
                if total_true_positives[c] == 0:
                    recall.append(0)
                else:
                    recall.append(running_corrects / total_true_positives[c])
            ap_to_class[c].append(voc_ap(recall, precision))

    logging.debug(f"Compute mAP time: {time.perf_counter() - start}")
    return ap_to_class


def evaluate_model(model: tf.keras.Model, generator: DataGenerator, iou_true_positive_threshold, nms_iou_threshold,
                   score_threshold, no_classes=len(ENCODE_LABEL)):
    """
    Computes mean average precision and average precisions for an entire dataset

    :param model: neural network under evaluation
    :param generator: generator for the dataset
    :param iou_true_positive_threshold: minimum IOU for a true positive
    :param nms_iou_threshold: maximum IOU for NMS
    :param score_threshold: minimum score a bounding box can have
    :param no_classes: number of classes
    :return: mAP, average precisions, number of bounding boxes detected for each class
    """
    average_precisions = []
    ap_to_class_all = {c: [] for c in range(no_classes)}
    for i in range(len(generator)):
        start_total = time.perf_counter()
        start = time.perf_counter()
        data, y_true = generator[i]
        logging.debug(f"Load data time: {time.perf_counter() - start}")
        y_pred = model.predict(data)
        start = time.perf_counter()

        ap_to_class = mean_average_precision(y_true, y_pred, generator.anchors, iou_true_positive_threshold,
                                             nms_iou_threshold,
                                             score_threshold, no_classes)

        logging.debug(f"time mAP: {time.perf_counter() - start}")
        logging.debug(f"{i + 1}/{len(generator)} time total: {time.perf_counter() - start_total}")
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


def log_to_file_map(model, generator, model_name, iou_true_positive_threshold, nms_iou_threshold, score_threshold):
    """
    Logs to a file the results

    :param model: neural network under evaluation
    :param generator: generator for the dataset
    :param model_name: the name of the model
    :param iou_true_positive_threshold: minimum IOU for a true positive
    :param nms_iou_threshold: maximum IOU for NMS
    :param score_threshold: minimum score a bounding box can have
    """
    dir = f"mean_average_precisions/iou_tp={iou_true_positive_threshold}"
    file = f"{dir}/iou_tp={iou_true_positive_threshold}_nms_iou={nms_iou_threshold}_score={score_threshold}.csv"
    if not os.path.exists(file):
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(file, "w") as f:
            f.write("model_name,mAP,Bus,Car,Vehicle registration plate,no_items\n")
    else:
        with open(file, "r") as f:
            for line in f.readlines()[1:]:
                if model_name in line:
                    logging.info(f"Already computed for {model_name}")
                    return
    mAP, aps, no_items = evaluate_model(model, generator, iou_true_positive_threshold=iou_true_positive_threshold,
                                        nms_iou_threshold=nms_iou_threshold,
                                        score_threshold=score_threshold)

    mAP = float(mAP)
    line = f"{model_name},{round(mAP, 4)},"
    for c in range(3):
        line += f"{round(aps[c], 4)},"
    line += f"{no_items}/{len(generator.image_paths)}\n".replace(",", " ")
    with open(file, "a") as f:
        f.write(line)


def compute_map_to_file(model, generator):
    iou_true_positive_thresholds = [0.3, 0.5, 0.7]
    nms_iou_thresholds = [0.3, 0.4, 0.5]
    score_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for iou_true_positive_threshold in iou_true_positive_thresholds:
        for nms_iou_threshold in nms_iou_thresholds:
            for score_threshold in score_thresholds:
                start = time.perf_counter()
                log_to_file_map(model, generator,
                                model_name=model_name,
                                iou_true_positive_threshold=iou_true_positive_threshold,
                                nms_iou_threshold=nms_iou_threshold,
                                score_threshold=score_threshold
                                )
                logging.info(f"Done with: {iou_true_positive_threshold} - {nms_iou_threshold} - {score_threshold} in {time.perf_counter()-start}")


def compute_single_map(model, generator, iou_true_positive_threshold, nms_iou_threshold, score_threshold):
    mAP, aps, no_items = evaluate_model(model, generator, iou_true_positive_threshold=iou_true_positive_threshold,
                                        nms_iou_threshold=nms_iou_threshold,
                                        score_threshold=score_threshold)
    print(f"mAP: {mAP}")
    print(aps)
    print(no_items)


if __name__ == '__main__':
    model_name = "model_v42_fine_tuned"
    #model_name = "coco_v3"
    model = tf.keras.models.load_model(f"weights/{model_name}.h5", custom_objects={
        'RandomColorAugmentation': RandomColorAugmentation,
        'Cutout': Cutout
    }, compile=False)
    generator = DataGenerator(PATH_TO_TEST, batch_size=8, shuffle=False)
    logging.Filter()
    compute_map_to_file(model, generator)
    #compute_single_map(model, generator, 0.5, 0.2, 0.4)

