from cv2 import cv2
from matplotlib import pyplot as plt

from settings import *
from utils import *


class Image:
    def __init__(self, dir=None, image_name=None, min_clip_val=0, max_clip_val=IMAGE_SIZE-1):
        self.bounding_boxes = []
        self.min_clip_val = min_clip_val
        self.max_clip_val = max_clip_val
        if dir and image_name:
            self.image_name = image_name
            self.test_dir = dir
            self.image = cv2.cvtColor(cv2.imread(f"{dir}\\{image_name}"), cv2.COLOR_BGR2RGB)
            self.read_bounding_boxes(f"{dir}\\Label\\{image_name[:-4]}.txt")

    def read_bounding_boxes(self, path):
        with open(path) as f:
            for line in f.readlines():
                tokens = line.split()
                label = tokens[0] if len(tokens) == 5 else " ".join(tokens[:len(tokens) - 4])
                coordinates = [my_round(float(t)) for t in tokens[-4:]]
                if coordinates[0] < coordinates[2]-1 and coordinates[1] < coordinates[3]-1:
                    self.bounding_boxes.append(BoundingBox(ENCODE_LABEL[label], *coordinates))
            self.clip_boxes(self.min_clip_val, self.max_clip_val)

    def with_bboxes(self, width=3):
        return with_bounding_boxes(self.image, self.bounding_boxes, width)

    def clip_boxes(self, min_val, max_val):
        for bbox in self.bounding_boxes:
            bbox.x_min = np.clip(bbox.x_min, min_val, max_val)
            bbox.x_max = np.clip(bbox.x_max, min_val, max_val)
            bbox.y_min = np.clip(bbox.y_min, min_val, max_val)
            bbox.y_max = np.clip(bbox.y_max, min_val, max_val)

    def shift_boxes(self, x, y):
        for bbox in self.bounding_boxes:
            bbox.x_min += x
            bbox.x_max += x
            bbox.y_min += y
            bbox.y_max += y

    def save_image(self, dir):
        if not self.image_name:
            raise Exception("Image name is not set")
        image_to_save = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{dir}/{self.image_name}", image_to_save)
        with open(f"{dir}\\Label\\{self.image_name[:-4]}.txt", "w") as f:
            for bbox in self.bounding_boxes:
                coordinates = " ".join([str(coord) for coord in [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]])
                f.write(f"{DECODE_LABEL[bbox.c]} {coordinates}\n")

    def bounding_boxes_as_arrays(self):
        return [b.as_coordinates_array() for b in self.bounding_boxes]

    def bounding_boxes_as_array_with_classes(self):
        return np.asarray([b.as_coordinates_array_with_class() for b in self.bounding_boxes])


class BoundingBox:
    def __init__(self, c, x_min, y_min, x_max, y_max, score=None):
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)
        self.c = int(c)
        self.score = score

    def width_height(self):
        return self.x_max - self.x_min, self.y_max - self.y_min

    def as_coordinates_array(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])

    def as_coordinates_array_with_class(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max, self.c])

    def center(self):
        return my_round((self.x_min + self.x_max) / 2), my_round((self.y_min + self.y_max) / 2)

    def rescale_box(self, new_w, new_h, original_w=IMAGE_SIZE, original_h=IMAGE_SIZE):
        new_x_min = (self.x_min/original_w)*new_w
        new_x_max = (self.x_max/original_w)*new_w
        new_y_min = (self.y_min/original_h)*new_h
        new_y_max = (self.y_max/original_h)*new_h
        return BoundingBox(self.c, new_x_min, new_y_min, new_x_max, new_y_max, self.score)

    def __str__(self):
        return str(self.as_coordinates_array_with_class())


def iou(bbox, other_bbox):
    """
    :param bbox: array of coordinates x_min, y_min, x_max, y_max
    :param other_bbox: array of coordinates x_min, y_min, x_max, y_max
    :return: Intersection/Union
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
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)
    print("iou = " + str(iou(box1, box2)))

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

    if USE_COCO:
        images = [
            Image("D:\\datasets\\coco\\coco_resized\\val", "000000000285.jpg", max_clip_val=10000),
            Image("D:\\datasets\\coco\\train2017", "000000000127.jpg", max_clip_val=10000),
            Image("D:\\datasets\\coco\\coco_resized\\train", "000000000127.jpg", max_clip_val=10000),
            Image("C:\\Users\\Dragos\\datasets\\coco\\train", "000000429913.jpg", max_clip_val=10000)
        ]
    else:
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
