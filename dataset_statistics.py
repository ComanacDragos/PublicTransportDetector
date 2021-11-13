import os
import sys
import threading
from multiprocessing import Pool

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
                currentClass = " ".join(tokens[:len(tokens)-4])
                if currentClass in classes:
                    classes[currentClass] += 1
                else:
                    classes[currentClass] = 1


def sizes():
    stages = {}
    for stage, path in paths.items():
        classes = {}
        files = [f"{path}/label/{file}" for file in os.listdir(path + '/label')]
        chunk = len(files) // os.cpu_count()
        threads = []
        print("stage", stage, len(files))
        for startIndex in range(0, os.cpu_count()):
            if startIndex == os.cpu_count() - 1:
                filesToProcess = files[startIndex * chunk:]
            else:
                filesToProcess = files[startIndex * chunk: (startIndex + 1) * chunk]
            thread = threading.Thread(target=process_annotations, args=[filesToProcess, classes])
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()
        stages[stage] = classes
    return stages


if __name__ == '__main__':
    for stage, classes in sizes().items():
        print(stage, sum(classes.values()))
        for currentClass, occurrences in classes.items():
            print(f"\t{currentClass} : {occurrences}")
