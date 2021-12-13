import threading
import time

from Image import Image
from utils import *

from settings import *

paths = {
    'train': PATH_TO_TRAIN,
    'validation': PATH_TO_VALIDATION,
    'test': PATH_TO_TEST
}


def process_annotations(files, classes):
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                tokens = line.split()
                currentClass = " ".join(tokens[:len(tokens) - 4])
                if currentClass in classes:
                    classes[currentClass] += 1
                else:
                    classes[currentClass] = 1


def sizes():
    stages = {}
    for stage, path in paths.items():
        classes = {}
        files = [f"{path}/label/{file}" for file in os.listdir(path + '/label')]
        print(path, len(files))
        run_task(files, process_annotations, [classes])
        stages[stage] = classes
    return stages


def max_labels_worker(images, dir, output, lock):
    local_output = []
    for path in images:
        img = Image(dir, path)
        local_output.append(len(img.bounding_boxes))
    m = max(local_output)
    lock.acquire()
    output.append(m)
    lock.release()


def get_max_labels_per_box():
    lock = threading.Lock()
    m = -1
    for dir in [PATH_TO_VALIDATION, PATH_TO_TEST, PATH_TO_TRAIN]:
        print(dir)
        images = os.listdir(dir)
        images.remove("Label")
        output = []
        run_task(images, max_labels_worker, [dir,output, lock])
        rez = max(output)
        if rez > m:
            m = rez
        print(rez)
    print(m)


if __name__ == '__main__':
    """for stage, classes in sizes().items():
        print(stage, sum(classes.values()))
        for currentClass, occurrences in classes.items():
            print(f"\t{currentClass} : {occurrences}")
    """
    get_max_labels_per_box()
