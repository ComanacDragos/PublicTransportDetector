import threading
import os
import tensorflow as tf

import numpy as np


def sum_over_axes(input_tensor, axes):
    x = input_tensor
    for a in reversed(axes):
        x = tf.reduce_sum(x, axis=a)
    return x


def my_round(a, threshold=0.5):
    return int(a) if a - int(a) < threshold else int(a) + 1


def run_task(items, target, args):
    """
    Splits items in several batches and runs target with args on each batch in parallel
    (on same number of threads as cpu cores)
    """
    chunk = len(items) // os.cpu_count()
    threads = []
    for startIndex in range(0, os.cpu_count()):
        if startIndex == os.cpu_count() - 1:
            filesToProcess = items[startIndex * chunk:]
        else:
            filesToProcess = items[startIndex * chunk: (startIndex + 1) * chunk]
        thread = threading.Thread(target=target, args=[filesToProcess] + args)
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()


def with_bounding_boxes(img, bounding_boxes, width):
    copy = np.copy(img)
    for bbox in bounding_boxes:
        color = [0, 0, 0]
        color[bbox.c % 3] = 255
        copy[bbox.y_min - width:bbox.y_min + width, bbox.x_min:bbox.x_max] = color
        copy[bbox.y_max - width:bbox.y_max + width, bbox.x_min:bbox.x_max] = color
        copy[bbox.y_min:bbox.y_max, bbox.x_min - width:bbox.x_min + width] = color
        copy[bbox.y_min:bbox.y_max, bbox.x_max - width:bbox.x_max + width] = color
    return copy
