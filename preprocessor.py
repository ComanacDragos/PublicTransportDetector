import os

import albumentations
import matplotlib.pyplot as plt

from Image import *


def resize_image(image: Image, h=IMAGE_SIZE, w=IMAGE_SIZE):
    """
    :param img_arr: original image as a numpy array
    :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :param h: resized height dimension of image
    :param w: resized weight dimension of image
    :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    """
    # create resize transform pipeline
    transform = albumentations.Compose(
        [albumentations.Resize(height=h, width=w, always_apply=True)],
        bbox_params=albumentations.BboxParams(format='pascal_voc'))

    transformed = transform(image=image.image, bboxes=image.bounding_boxes_as_array_with_classes())
    image.image = np.asarray(transformed["image"])

    boxes = []

    for boxArray in transformed["bboxes"]:
        boxes.append(BoundingBox(boxArray[-1], *[my_round(coord) for coord in boxArray[:4]]))
    image.bounding_boxes = boxes


def preprocess_dataset_worker(image_files, dir, new_dir):
    for image_file in image_files:
        image = Image(dir, image_file)
        resize_image(image)
        image.save_image(new_dir)


def preprocess_dataset():
    dirs = {
        PATH_TO_TRAIN_UNPROCESSED: PATH_TO_TRAIN,
        PATH_TO_VALIDATION_UNPROCESSED: PATH_TO_VALIDATION,
        PATH_TO_TEST_UNPROCESSED: PATH_TO_TEST,
    }

    for dir, new_dir in dirs.items():
        print(new_dir, " starting...")
        image_files = os.listdir(dir)
        image_files.remove("Label")
        run_task(image_files, preprocess_dataset_worker, [dir, new_dir])


def test_resize():
    paths = os.listdir(PATH_TO_VALIDATION_UNPROCESSED)
    paths.remove("Label")
    print(len(paths))
    np.random.shuffle(paths)
    image = Image(PATH_TO_VALIDATION_UNPROCESSED, paths[0])
    print(image.image.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(image.with_bboxes(5))
    resize_image(image)
    print(image.image.shape)
    plt.subplot(1, 2, 2)
    plt.imshow(image.with_bboxes(5))
    plt.show()


if __name__ == '__main__':
    preprocess_dataset()