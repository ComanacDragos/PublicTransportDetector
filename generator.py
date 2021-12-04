import time

import numpy as np
import tensorflow as tf
from Image import *


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, batch_size, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), anchors_path="data/anchors.pickle", shuffle=True):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.anchors = process_anchors(anchors_path)

        self.image_paths = []
        self.db_dir = db_dir
        self.get_data()
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def get_data(self):
        """"
        Loads the paths to the images from the database directory
        """
        self.image_paths = os.listdir(self.db_dir)
        self.image_paths.remove("Label")

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for i in range(len(batch_indices)):
            index = batch_indices[i]
            image = Image(self.db_dir, self.image_paths[index])
            batch_x.append(image.image)
            batch_y.append(generate_output_array(image, self.anchors))
        # batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        # if required, shuffle your data after each epoch
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)


def process_anchors(path):
    anchor_boxes = np.load(path, allow_pickle=True)
    anchors = []
    for i in range(anchor_boxes.shape[0]):
        x_min, y_min, x_max, y_max = anchor_boxes[i]
        anchors.append([x_max - x_min, y_max - y_min])
    return np.array(anchors)


def clip_value(value):
    return max(0, min(value, IMAGE_SIZE - 1))


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
    print("=" * 20)
    for b in boxes:
        print(b.c, b.as_coordinates_array())


def test_generator():
    generator = DataGenerator(PATH_TO_VALIDATION, 32, (IMAGE_SIZE, IMAGE_SIZE, 3), "data/anchors.pickle")
    start = time.time()
    data, labels = generator[0]
    print("Time to load: ", time.time() - start)
    print(len(generator), len(generator.image_paths))
    last_data, last_labels = generator[len(generator)-1]
    print(data.shape, last_data.shape)
    print(labels.shape, last_labels.shape)

    boxes = interpret_output(labels[0], generator.anchors)
    image = with_bounding_boxes(data[0], boxes, 3, (255, 0, 0))

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    print("starting...")
    # test_generate_output_array()
    test_generator()
