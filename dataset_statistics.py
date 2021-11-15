import time

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


if __name__ == '__main__':
    for stage, classes in sizes().items():
        print(stage, sum(classes.values()))
        for currentClass, occurrences in classes.items():
            print(f"\t{currentClass} : {occurrences}")

