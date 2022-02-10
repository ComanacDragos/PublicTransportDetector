import time

import numpy as np

from image import *
from data_augmentation import mosaic


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, batch_size=BATCH_SIZE, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                 anchors_path=ANCHORS_PATH,
                 shuffle=True, limit_batches=None):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.anchors = process_anchors(anchors_path)

        self.image_paths = []
        self.db_dir = db_dir
        self.get_data(limit_batches)
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def get_data(self, limit_batches):
        """"
        Loads the paths to the images from the database directory
        """
        self.image_paths = os.listdir(self.db_dir)
        self.image_paths.remove("Label")
        if limit_batches is not None:
            self.image_paths = self.image_paths[:limit_batches * self.batch_size]

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
        batch_true_boxes = []
        for i in range(len(batch_indices)):
            indices = np.random.choice(batch_indices, 4, replace=False)
            image = mosaic(
                [Image(self.db_dir, self.image_paths[idx]) for idx in indices]
            )
            #index = batch_indices[i]
            #image = Image(self.db_dir, self.image_paths[index])
            output, true_boxes = generate_output_array(image, self.anchors)
            batch_x.append(image.image)
            batch_true_boxes.append(true_boxes)
            batch_y.append(output)
        return [np.asarray(batch_x), np.asarray(batch_true_boxes)], np.asarray(batch_y)

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
    if anchor_boxes.shape[-1] == 2:
        return anchor_boxes
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
    - tw, th the offsets with respect to the anchor
    - C number of classes
    """
    no_anchors = anchors.shape[0]
    output = np.zeros((GRID_SIZE, GRID_SIZE, no_anchors, (5 + len(ENCODE_LABEL))))
    cell_size = IMAGE_SIZE / GRID_SIZE
    true_boxes = np.zeros((1, 1, 1, MAX_BOXES_PER_IMAGES, 4))
    for box_index, bbox in enumerate(image.bounding_boxes):
        x_center, y_center = bbox.center()
        cx = int(x_center / cell_size)
        cy = int(y_center / cell_size)

        tx = x_center / cell_size  # (x_center - cx * cell_size) / cell_size + cx
        ty = y_center / cell_size  # (y_center - cy * cell_size) / cell_size + cy

        best_anchor = -1
        best_iou = -1
        for i in range(no_anchors):
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

        # anchor_width, anchor_height = anchors[best_anchor]
        box_width, box_height = bbox.width_height()
        tw = box_width  # np.log(box_width) - np.log(anchor_width)
        th = box_height  # np.log(box_height) - np.log(anchor_height)

        output[cy, cx, best_anchor, 0] = tx
        output[cy, cx, best_anchor, 1] = ty
        output[cy, cx, best_anchor, 2] = tw
        output[cy, cx, best_anchor, 3] = th
        output[cy, cx, best_anchor, 4] = 1.
        output[cy, cx, best_anchor, 5 + bbox.c] = 1.

        true_boxes[0, 0, 0, box_index] = [tx, ty, tw, th]
    return output, true_boxes


def interpret_ground_truth(output, anchors, obj_threshold=0.5):
    cell_size = IMAGE_SIZE / GRID_SIZE
    boxes = []
    for cx in range(GRID_SIZE):
        for cy in range(GRID_SIZE):
            for i, (anchor_width, anchor_height) in enumerate(anchors):
                if output[cx, cy, i, 4] >= obj_threshold:
                    tx, ty, tw, th = output[cx, cy, i, :4]
                    print(tx, ty, tw, th)
                    bx = tx * cell_size
                    by = ty * cell_size
                    bw = tw #anchor_width * np.exp(tw)
                    bh = th #anchor_height * np.exp(th)
                    x_min, y_min = bx - bw / 2, by - bh / 2
                    x_max, y_max = bx + bw / 2, by + bh / 2
                    boxes.append(BoundingBox(
                        np.argmax(output[cx, cy, i, 5:]),
                        my_round(x_min, 0.51), my_round(y_min, 0.51), my_round(x_max, 0.51), my_round(y_max, 0.51)
                    ))
    return boxes


# 0000599864fd15b3
def test_generator():
    generator = DataGenerator(PATH_TO_TRAIN, 32, (IMAGE_SIZE, IMAGE_SIZE, 3), shuffle=False)
    start = time.time()
    ground_truth, labels = generator[0]
    print("Time to load: ", time.time() - start)
    data, true_boxes = ground_truth[0], ground_truth[1]
    print(len(generator), len(generator.image_paths))
    ground_truth, last_labels = generator[len(generator) - 1]
    last_data, last_true_boxes = ground_truth[0], ground_truth[1]
    print(data.shape, last_data.shape, true_boxes.shape)
    print(labels.shape, last_labels.shape, last_true_boxes.shape)

    for i in range(5):
        box = true_boxes[0, 0, 0, 0, i, :]
        if np.sum(box) == 0:
            break
        print(box)

    for cx in range(GRID_SIZE):
        for cy in range(GRID_SIZE):
            for a in range(len(generator.anchors)):
                anchor = labels[0, cx, cy, a, :]
                if anchor[4] == 1:
                    print(anchor, cx, cy)
                else:
                    if np.sum(anchor) != 0:
                        print("Error!!")

    boxes = interpret_ground_truth(labels[0], generator.anchors)
    print([b.as_coordinates_array() for b in boxes])
    image = with_bounding_boxes(data[0], boxes, 3)

    plt.imshow(image)
    plt.show()


def visualize_images(no_images=4):
    generator = DataGenerator(PATH_TO_TRAIN, no_images, (IMAGE_SIZE, IMAGE_SIZE, 3))
    [images, _], ground_truth = generator[0]
    print(np.asarray(images).shape)
    fig, axs = plt.subplots(no_images, 1, figsize=(10, 10))
    for i in range(0, no_images):
        plt.subplot(no_images, 1, i+1)
        img = Image()
        img.image = images[i]
        img.bounding_boxes = interpret_ground_truth(ground_truth[i], generator.anchors)
        plt.imshow(img.with_bboxes())
    plt.show()

if __name__ == '__main__':
    print("starting...")
    #test_generator()
    #print(5/2)
    visualize_images()
