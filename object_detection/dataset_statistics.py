from image import Image
from settings import *
from utils import *

paths = {
    'train': PATH_TO_TRAIN,
    'validation': PATH_TO_VALIDATION,
    'test': PATH_TO_TEST
}


def process_annotations(files, classes, no_files):
    for file in files:
        contains_class = {}
        with open(file) as f:
            for line in f.readlines():
                tokens = line.split()
                currentClass = " ".join(tokens[:len(tokens) - 4])
                contains_class[currentClass] = True
                if currentClass in classes:
                    classes[currentClass] += 1
                else:
                    classes[currentClass] = 1
        for c, contains in contains_class.items():
            if contains:
                if c in no_files:
                    no_files[c] += 1
                else:
                    no_files[c] = 1


def sizes():
    stages = {}
    for stage, path in paths.items():
        classes = {}
        no_files = {}
        files = [f"{path}/label/{file}" for file in os.listdir(path + '/label')]
        print(path, len(files))
        run_task(files, process_annotations, [classes, no_files])
        stages[stage] = classes, no_files
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
    for stage, (classes, no_files) in sizes().items():
        print(stage, sum(classes.values()))
        print("\tBounding boxes:")
        for currentClass, occurrences in classes.items():
            print(f"\t\t{currentClass} : {occurrences}")
        print("\tNumber of files:")
        for currentClass, occurrences in no_files.items():
            print(f"\t\t{currentClass} : {occurrences}")
    get_max_labels_per_box()
