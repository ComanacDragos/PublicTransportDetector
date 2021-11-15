import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from settings import *
from utils import *


class Image:
    def __init__(self, dir, image_name):
        self.image = cv2.cvtColor(cv2.imread(f"{dir}\\{image_name}"), cv2.COLOR_BGR2RGB)
        self.bounding_boxes = []
        self.read_bounding_boxes(f"{dir}\\Label\\{image_name[:-4]}.txt")
        self.image = cv2.resize(self.image, (IMAGE_SIZE, IMAGE_SIZE))

    def read_bounding_boxes(self, path):
        with open(path) as f:
            for line in f.readlines():
                tokens = line.split()
                label = tokens[0] if len(tokens) == 5 else " ".join(tokens[:len(tokens) - 4])
                height = self.image.shape[0]
                width = self.image.shape[1]

                coordinates = [my_round(float(t)) for t in tokens[-4:]]
                coordinates[0] = int(IMAGE_SIZE / width * coordinates[0])
                coordinates[1] = int(IMAGE_SIZE / height * coordinates[1])
                coordinates[2] = int(IMAGE_SIZE / width * coordinates[2])
                coordinates[3] = int(IMAGE_SIZE / height * coordinates[3])

                self.bounding_boxes.append(BoundingBox(ENCODE_LABEL[label], *coordinates))

    def with_bboxes(self, width):
        img = self.image
        for bbox in self.bounding_boxes:
            color = [200, 0, 0]
            color[bbox.c % 3] = 255
            img[bbox.y_min - width:bbox.y_min + width, bbox.x_min:bbox.x_max] = color
            img[bbox.y_max - width:bbox.y_max + width, bbox.x_min:bbox.x_max] = color
            img[bbox.y_min:bbox.y_max, bbox.x_min - width:bbox.x_min + width] = color
            img[bbox.y_min:bbox.y_max, bbox.x_max - width:bbox.x_max + width] = color
        return img

    def bounding_boxes_as_arrays(self):
        return [b.as_coordinates_array() for b in self.bounding_boxes]


class BoundingBox:
    def __init__(self, c, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.c = c

    def width_height(self):
        return self.x_max - self.x_min, self.y_max - self.y_min

    def as_coordinates_array(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])

    def center(self):
        return my_round((self.x_min + self.x_max) / 2), my_round((self.y_min + self.y_max) / 2)


def iou(bbox, other_bbox):
    """
    :param bbox: array of coordinates x_min, y_min, x_max, y_max
    :param other_bbox: array of coordinates x_min, y_min, x_max, y_max
    :return: IOU
    """
    x_min, y_min, x_max, y_max = bbox
    other_x_min, other_y_min, other_x_max, other_y_max = other_bbox

    intersect_width = max(min(x_max, other_x_max) - max(x_min, other_x_min), 0)
    intersect_height = max(min(y_max, other_y_max) - max(y_min, other_y_min), 0)

    bbox_width, bbox_height = x_max - x_min, y_max - y_min
    other_bbox_width, other_bbox_height = other_x_max - other_x_min, other_y_max - other_y_min

    intersect = intersect_height * intersect_width
    union = bbox_height * bbox_width + other_bbox_height * other_bbox_width - intersect
    return float(intersect) / union


def iou_bbox(bbox, other_bbox):
    return iou(bbox.as_coordinates_array(), other_bbox.as_coordinates_array())


if __name__ == '__main__':
    bbox = BoundingBox(-1, 1, 1, 5, 5)
    other_bbox = BoundingBox(-1, 3, 3, 6, 6)
    print(iou_bbox(bbox, other_bbox))

    bbox = BoundingBox(-1, 1, 1, 5, 5)
    other_bbox = BoundingBox(-1, 2, 2, 4, 4)
    print(iou_bbox(bbox, other_bbox))

    bbox = BoundingBox(-1, 1, 1, 5, 5)
    other_bbox = BoundingBox(-1, 20, 20, 40, 40)
    print(iou_bbox(bbox, other_bbox))

    bbox = BoundingBox(-1, 1, 1, 5, 5)
    other_bbox = BoundingBox(-1, 1, 1, 5, 5)
    print(iou_bbox(bbox, other_bbox))

    print(BoundingBox(-1, 3, 3, 6, 6).center())

    images = [
        Image(PATH_TO_VALIDATION, "0f4bfc46402a9f52.jpg"),
        Image(PATH_TO_VALIDATION, "3f10138eb086f2e9.jpg"),
        Image(PATH_TO_VALIDATION, "3fc5aee11ddb3651.jpg"),
        Image(PATH_TO_VALIDATION, "4a23eee283f294b6.jpg"),
    ]

    plt.figure(figsize=(10, 20))
    for i, img in enumerate(images):
        plt.subplot(len(images), 1, i + 1)
        plt.imshow(img.with_bboxes(3))
    plt.show()
