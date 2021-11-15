import time

import numpy as np

from Image import *


def _worker(paths, boxes, dir, mutex):
    local_boxes = []
    for path in paths:
        image_boxes = Image(dir, path).bounding_boxes_as_arrays()
        local_boxes += image_boxes
    mutex.acquire()
    boxes += local_boxes
    mutex.release()


def read_boxes():
    directories = [PATH_TO_TRAIN, PATH_TO_TEST, PATH_TO_VALIDATION]
    boxes = []
    mutex = threading.Lock()

    for dir in directories:
        images = os.listdir(dir)
        images.remove("Label")
        images = images
        print(dir, len(images))
        run_task(images, _worker, [boxes, dir, mutex])
    boxes = np.array(boxes)
    return boxes


def distance(centroid, box):
    return 1 - iou(centroid, box)


def generate_centroid():
    x = np.random.randint(0, IMAGE_SIZE, 2)
    y = np.random.randint(0, IMAGE_SIZE, 2)
    return np.min(x), np.min(y), np.max(x), np.max(y)


def generate_anchors(bounding_boxes=None, prior_centroids=None):
    if not bounding_boxes:
        boxes = read_boxes()
    else:
        boxes = bounding_boxes
    no_boxes = boxes.shape[0]
    if prior_centroids is None:
        centroids = set()
        while len(centroids) != ANCHORS:
            new_centroid = generate_centroid()
            add = True
            for centroid in centroids:
                add = add and iou(centroid, new_centroid) < 0.5
            if add:
                centroids.add(new_centroid)
        centroids = np.array(list(centroids))
    else:
        centroids = prior_centroids

    print(boxes.shape, centroids.shape)
    old_assignments = np.ones(no_boxes) * -1
    old_distances = np.ones((no_boxes, ANCHORS)) * -1
    change = True
    iteration = 1
    while change:
        distances = []
        for i in range(no_boxes):
            centroid_distances = []
            for j in range(len(centroids)):
                centroid_distances.append(distance(centroids[j], boxes[i]))
            distances.append(centroid_distances)
        distances = np.array(distances)

        assignments = np.argmin(distances, axis=1)

        print(iteration, np.sum(np.abs(old_distances - distances)), no_boxes - np.sum((assignments == old_assignments)))
        iteration += 1

        if (assignments == old_assignments).all():
            return centroids

        old_assignments = assignments.copy()
        old_distances = distances.copy()
        for i in range(len(centroids)):
            indices = np.nonzero((assignments == i))[0]
            if len(indices) != 0:
                centroids[i] = np.sum(boxes[indices], axis=0) / len(indices)
            else:
                print("! empty cluster...restarting...")
                return generate_anchors(boxes, prior_centroids)


if __name__ == '__main__':
    start = time.time()

    anchors = generate_anchors(prior_centroids=np.load("data/anchors_10000.pickle", allow_pickle=True))
    print(anchors)
    anchors.dump("data/anchors.pickle")

    print("Time to generate anchors: ", time.time() - start)

    anchors = np.load("data/anchors.pickle", allow_pickle=True)
    print(anchors)

    """
    [[114 130 294 280]
     [318 117 396 207]
     [ 15 130  86 221]
     [ 65 282 169 353]
     [295 266 376 339]]
    """
