import random
import time

from image import *


def _worker(paths, boxes, dir, mutex):
    """
    Multi-threaded worker for reading the bounding boxes

    :param paths: list composed of names of images
    :param boxes: global variable holding all boxes
    :param dir: the directory which holds the images
    :param mutex: lock for accessing the boxes global variable
    """
    local_boxes = []
    for path in paths:
        image_boxes = Image(dir, path).bounding_boxes_as_arrays()
        local_boxes += image_boxes
    mutex.acquire()
    boxes += local_boxes
    mutex.release()


def read_boxes():
    """
    Reads all bounding boxes in the dataset using multiple threads

    :return: list of all bounding boxes
    """
    directories = [
        PATH_TO_TRAIN,
        #PATH_TO_TEST,
        PATH_TO_VALIDATION]
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
    """
    Computes the distance between a centroid and a bounding box (the center of the centroid is the center of the box)

    :param centroid: pair of width and height
    :param box: bounding box
    :return: 1-IOU(centroid, box)
    """
    x, y = BoundingBox(-1, *box).center()
    w, h = centroid
    return 1 - iou([x-w//2, y-h//w, x+w//2, y+h//2], box)


def generate_centroid():
    """
    Generates a random width and height that make a centroid

    :return: random width, height pair, between 1 and IMAGE_SIZE global parameter
    """
    width = random.randint(1, IMAGE_SIZE)
    height = random.randint(1, IMAGE_SIZE)
    return width, height


def generate_anchors(no_anchors, bounding_boxes=None, prior_centroids=None):
    """
    K-Means algorithm

    :param no_anchors: the number of desired anchors/centroids
    :param bounding_boxes: the bounding boxes on which K-Means is applied, if None, then the boxes are read
    :param prior_centroids: previously computed centroids, if None, they are generated randomly from scratch
    :return: list of generated anchors
    """
    if not bounding_boxes:
        boxes = read_boxes()
    else:
        boxes = bounding_boxes
    no_boxes = boxes.shape[0]
    if prior_centroids is None:
        centroids = set()
        while len(centroids) != no_anchors:
            new_centroid = generate_centroid()
            new_centroid_aspect_ratio = new_centroid[0] / new_centroid[1]
            add = True
            for centroid in centroids:
                add = add and abs(centroid[0]/centroid[1] - new_centroid_aspect_ratio) < 0.3
            if add:
                centroids.add(new_centroid)
        centroids = np.array(list(centroids))
    else:
        centroids = prior_centroids

    print(boxes.shape, centroids.shape)
    old_assignments = np.ones(no_boxes) * -1
    old_distances = np.ones((no_boxes, no_anchors)) * -1
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
                x_min, y_min, x_max, y_max = np.sum(boxes[indices], axis=0) / len(indices)
                centroids[i][0], centroids[i][1] = x_max - x_min, y_max - y_min
            else:
                print("! empty cluster...restarting...")
                return generate_anchors(boxes, prior_centroids)


def run_generate_anchors(no_anchors):
    """
    Driver function for running K-Means and saving the results

    :param no_anchors: the number of desired anchors
    """
    start = time.time()

    anchors = generate_anchors(no_anchors=no_anchors)
    print(anchors)
    anchors.dump(f"data/coco_anchors_{no_anchors}.pickle")

    print("Time to generate anchors: ", time.time() - start)

    anchors = np.load(f"data/coco_anchors_{no_anchors}.pickle", allow_pickle=True)
    print(anchors)


def visualize_anchors(path):
    anchors = np.load(path, allow_pickle=True)
    print(anchors)
    x_center = IMAGE_SIZE//2
    y_center = IMAGE_SIZE//2
    boxes = [BoundingBox(0,
                         x_center - width//2, y_center - height//2,
                         x_center + width//2, y_center + height//2) for width, height in anchors]
    for b in boxes:
        print(b.as_coordinates_array())
    img = with_bounding_boxes(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)), boxes, 1).astype(int)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    run_generate_anchors(5)

    visualize_anchors("data/coco_anchors_5.pickle")

    """
    [[114 130 294 280]
     [318 117 396 207]
     [ 15 130  86 221]
     [ 65 282 169 353]
     [295 266 376 339]]
     
     [[ 62  57]
     [276 246]
     [ 18  15]]
     
     [[ 84  79]
     [ 16  14]
     [  8   6]
     [ 36  30]
     [294 260]]

    [[ 10   8]
     [288 256]
     [ 76  70]
     [ 28  24]]
     
     COCO:
     
     [[ 64  92]
 [205 234]
 [  5  10]
 [ 12  18]
 [ 26  38]]
    """
