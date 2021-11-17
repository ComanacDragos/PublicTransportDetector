import time

import numpy as np
import tensorflow as tf
from Image import *


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, batch_size, input_shape, anchors_path, shuffle=True):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.anchors = process_anchors(anchors_path)

        self.data, self.labels = self.get_data(db_dir)
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def get_data(self, root_dir):
        """"
        Loads the paths to the images and their corresponding labels from the database directory
        """
        self.data = []
        self.labels = []

        image_paths = os.listdir(root_dir)
        image_paths.remove("Label")

        mutex = threading.Lock()
        run_task(image_paths, _read_images_worker, [root_dir, self.data, self.labels, self.anchors, mutex])
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        return self.data, self.labels

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = self.data[batch_indices]
        batch_y = self.labels[batch_indices]
        # batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return batch_x, batch_y

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        # if required, shuffle your data after each epoch
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)


def _read_images_worker(paths, dir, data, labels, anchors, mutex):
    local_data = []
    local_labels = []
    for path in paths:
        image = Image(dir, path)
        local_data.append(image.image)
        local_labels.append(generate_output_array(image, anchors))
    mutex.acquire()
    data += local_data
    labels += local_labels
    mutex.release()


def process_anchors(path):
    anchor_boxes = np.load(path, allow_pickle=True)
    anchors = []
    for i in range(anchor_boxes.shape[0]):
        x_min, y_min, x_max, y_max = anchor_boxes[i]
        anchors.append([x_max - x_min, y_max - y_min])
    return np.array(anchors)


def clip_value(value):
    return max(0, min(value, IMAGE_SIZE-1))


def generate_output_array(image: Image, anchors):
    """
    each example is mapped to the following:
    array of shape C x C x ANCHORS x (tx + ty + tw + th + obj_score + C)
    - tx, ty the offsets with respect to the grid cell
    - tw, th the offesets with respect to the anchor
    - C number of classes
    """
    output = np.zeros((GRID_SIZE, GRID_SIZE, ANCHORS, (5 + len(ENCODE_LABEL))))
    downsample_factor = IMAGE_SIZE / GRID_SIZE
    for bbox in image.bounding_boxes:
        x_center, y_center = bbox.center()
        cx = int(x_center / downsample_factor)
        cy = int(y_center / downsample_factor)

        tx = (x_center - cx * downsample_factor) / downsample_factor
        ty = (y_center - cy * downsample_factor) / downsample_factor

        best_anchor = -1
        best_iou = -1
        for i in range(ANCHORS):
            anchor_width, anchor_height = anchors[i]
            x_min, y_min = x_center - anchor_width // 2, y_center - anchor_height // 2
            x_max, y_max = x_center + anchor_width // 2, y_center + anchor_height // 2

            x_min = clip_value(x_min)
            y_min = clip_value(y_min)
            x_max = clip_value(x_max)
            y_max = clip_value(y_max)

            current_iou = iou(bbox.as_coordinates_array(), [x_min, y_min, x_max, y_max])

            if current_iou > best_iou:
                best_anchor = i
                best_iou = current_iou

        anchor_width, anchor_height = anchors[best_anchor]
        box_width, box_height = bbox.width_height()
        tw = np.log(box_width) - np.log(anchor_width)
        th = np.log(box_height) - np.log(anchor_height)

        output[cx, cy, best_anchor, 0] = tx
        output[cx, cy, best_anchor, 1] = ty
        output[cx, cy, best_anchor, 2] = tw
        output[cx, cy, best_anchor, 3] = th
        output[cx, cy, best_anchor, 4] = 1.0
        output[cx, cy, best_anchor, 5 + bbox.c] = 1
    return output


def interpret_output(output, anchors):
    downsample_factor = IMAGE_SIZE / GRID_SIZE
    boxes = []
    for cx in range(GRID_SIZE):
        for cy in range(GRID_SIZE):
            for i, (anchor_width, anchor_height) in enumerate(anchors):
                if output[cx, cy, i, 4] >= OBJ_THRESHOLD:
                    tx, ty, tw, th = output[cx, cy, i, :4]
                    bx = (tx + cx) * downsample_factor
                    by = (ty + cy) * downsample_factor
                    bw = anchor_width * np.exp(tw)
                    bh = anchor_height * np.exp(th)
                    x_min, y_min = bx - bw / 2, by - bh / 2
                    x_max, y_max = bx + bw / 2, by + bh / 2
                    boxes.append(BoundingBox(
                        np.argmax(output[cx, cy, i, 5:]),
                        my_round(x_min, 0.51), my_round(y_min, 0.51), my_round(x_max, 0.51), my_round(y_max, 0.51)
                    ))
    return boxes


def test_generate_output_array():
    anchors = process_anchors("data/anchors.pickle")
    # print(anchors)

    originalImage = Image(PATH_TO_VALIDATION, "4a23eee283f294b6.jpg")
    output = generate_output_array(originalImage, anchors)
    boxes = interpret_output(output, anchors)

    for b in originalImage.bounding_boxes:
        print(b.c, b.as_coordinates_array())
    print("="*20)
    for b in boxes:
        print(b.c, b.as_coordinates_array())


def test_generator():
    start = time.time()
    generator = DataGenerator(PATH_TO_VALIDATION, 32, (IMAGE_SIZE, IMAGE_SIZE, 3), "data/anchors.pickle")
    print("Time to load: ", time.time() - start)
    print(generator.data.shape)
    print(generator.labels.shape)
    print(len(generator) * generator.batch_size)
    # for i in range(len(generator)):
    #    data, labels = generator[i]
    #    print(data.shape, labels.shape)
    #    print(np.min(labels), np.max(labels))


if __name__ == '__main__':
    # test_generate_output_array()
    test_generator()