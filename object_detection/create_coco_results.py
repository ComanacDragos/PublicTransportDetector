import numpy as np

from inference import *
from utils_file import *


def box_to_result_format(box: BoundingBox, image_id, category_to_id):
    x, y = box.center()
    w, h = box.width_height()

    return {
        "image_id": image_id,
        "category_id": category_to_id[DECODE_LABEL[box.c]],
        "bbox": [x, y, w, h],
        "score": round(float(box.score), 4)
    }


def create_results(split, score):
    if split == 'test-dev':
        PATH_TO_DIR = PATH_TO_TEST
    elif split == 'val':
        PATH_TO_DIR = PATH_TO_VALIDATION
    else:
        raise Exception('Invalid split')
    IMAGE_INFO = f"D:\\datasets\\coco\\annotations\\id_to_file_name_{split}.pickle"

    id_to_file = open_pickle(IMAGE_INFO)

    files = os.listdir(PATH_TO_DIR)
    files.remove("Label")
    for file in id_to_file.values():
        if file not in files:
            raise Exception(f"Not existent file {file}")

    model = tf.keras.models.load_model('weights/coco_v2.h5', custom_objects={
        'RandomColorAugmentation': RandomColorAugmentation,
        'Cutout': Cutout
    }, compile=False)

    id_to_category = open_json("D:\\datasets\\coco\\categories.json")
    category_to_id = {v: int(k) for k, v in id_to_category.items()}

    results = []
    for i, (id, file) in enumerate(id_to_file.items()):
        if i % 100 == 0:
            print(f"{i}/{len(id_to_file)}")
        bboxes = image_to_bounding_boxes(model, f"{PATH_TO_DIR}/{file}", score_threshold=score)
        image_results = [box_to_result_format(box, id, category_to_id) for box in bboxes]
        results += image_results
    print(results)
    to_json(results, f"coco_results/{split}_results_score={str(int(score*100))}.json")


def filter_results(new_score, results):
    new_results = []
    for result in results:
        if result['score'] >= new_score:
            new_results.append(result)
    return new_results


def statistics(file, path_to_dir):
    results = open_json(file)

    d = {}
    scores = []
    for result in results:
        category = result['category_id']
        if category in d:
            d[category] += 1
        else:
            d[category] = 1
        scores.append(result['score'])

    for k, v in d.items():
        print(k, v)
    print()

    print(len(scores), len(os.listdir(path_to_dir)) - 1, len(scores) / (len(os.listdir(path_to_dir)) - 1))
    print("mean", np.mean(scores))

    scores = np.sort(scores)

    quarter = len(scores) // 10
    print("lower mean", np.mean(scores[:quarter]))
    print("upper mean", np.mean(scores[len(scores) - quarter:]))


if __name__ == '__main__':
    split = "test-dev"
    score = 0.9

    results = open_json(f"coco_results/{split}_results_score=30.json")
    file = f"coco_results/{split}_results_score={str(int(score*100))}.json"
    to_json(filter_results(score, results), file)
    statistics(file, PATH_TO_TEST)
