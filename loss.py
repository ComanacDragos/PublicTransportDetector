from settings import *
import tensorflow as tf
import numpy as np


def create_cell_grid(no_anchors):
    x_offset = np.zeros((GRID_SIZE, GRID_SIZE))
    y_offset = np.zeros((GRID_SIZE, GRID_SIZE))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x_offset[i][j] = i
            y_offset[i][j] = j
    x_offset = np.stack([x_offset] * no_anchors, axis=-1)
    y_offset = np.stack([y_offset] * no_anchors, axis=-1)
    return tf.cast(tf.stack([x_offset, y_offset], -1), tf.float32)


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, anchors, true_boxes,
                 l_coord=1., l_noobj=0.5, l_class=1., l_obj=1.,
                 nb_class=3, iou_threshold=0.6, enable_logs=False):
        super(YoloLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_class = l_class
        self.l_obj = l_obj
        self.iou_threshold = iou_threshold

        self.anchors = anchors
        self.true_boxes = true_boxes
        self.cell_size = IMAGE_SIZE / GRID_SIZE

        self.cell_grid = create_cell_grid(len(anchors))

        self.nb_box = len(anchors) // 2
        self.enable_logs = enable_logs
        self.nb_class = nb_class
        self.class_wt = np.ones(self.nb_class, dtype="float32")

    def call(self, y_true, y_pred: tf.Tensor):
        """
        Each anchor is composed of 8 values:
        0, 1: x, y position
        2, 3: width, height
        4: if there is an object
        5, 6, 7: probabilities

        y_true, y_pred : shape -> (batch_size, grid_size, grid_size, anchors, 8)
        https://github.com/experiencor/keras-yolo2/blob/master/frontend.py
        """
        mask_shape = tf.shape(y_true)[:4]

        conf_mask = tf.zeros(mask_shape)

        # seen = tf.Variable(0.)
        # total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + self.cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * self.anchors  # np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.l_coord

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.cast(best_ious < 0.6, tf.float32) * (1 - y_true[..., 4]) * self.l_noobj

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.l_obj

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.l_class

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32))
        nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0.0, tf.float32))
        nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))
        # print((nb_coord_box + 1e-6) / 2.)
        loss_xy = tf.cast(tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask), tf.float32) / (
                nb_coord_box + 1e-6) / 2. / GRID_SIZE
        loss_wh = tf.cast(tf.reduce_sum(tf.square(tf.sqrt(true_box_wh) - tf.sqrt(pred_box_wh)) * coord_mask), tf.float32) / (
                nb_coord_box + 1e-6) / 2. / IMAGE_SIZE
        loss_conf = tf.cast(tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask),
                            tf.float32) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class
        if self.enable_logs:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(
                tf.compat.v1.to_float(true_box_conf > 0.5) * tf.compat.v1.to_float(pred_box_conf > 0.3))

            # current_recall = nb_pred_box / tf.cast(nb_true_box + 1e-6, tf.float32)
            # total_recall = tf.compat.v1.assign_add(total_recall, current_recall)

            loss = tf.compat.v1.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            # loss = tf.compat.v1.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            # loss = tf.compat.v1.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

        return loss


if __name__ == '__main__':
    YoloLoss(np.ones((3, 2)), None)

