import numpy as np
import matplotlib.pyplot as plt

from inference import *
from utils_file import *


def box_to_result_format(box: BoundingBox, image_id, category_to_id):
    return box_to_result_format_exact_category(box, image_id, category_to_id[DECODE_LABEL[box.c]])


def box_to_result_format_exact_category(box: BoundingBox, image_id, category_id):
    x, y = box.center()
    w, h = box.width_height()

    return {
        "image_id": image_id,
        "category_id": category_id,
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
    to_json(results, f"coco_results/{split}_results_score={str(int(score * 100))}.json")


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


def convert_bounding_boxes(split, score):
    if split == 'test-dev':
        PATH_TO_DIR = PATH_TO_TEST
        PATH_TO_DIR_UNPROCESSED = PATH_TO_TEST_UNPROCESSED
    elif split == 'val':
        PATH_TO_DIR = PATH_TO_VALIDATION
        PATH_TO_DIR_UNPROCESSED = PATH_TO_VALIDATION_UNPROCESSED
    else:
        raise Exception('Invalid split')
    results = open_json(f"coco_results/{split}_results_score={str(int(score * 100))}.json")
    IMAGE_INFO = f"D:\\datasets\\coco\\annotations\\id_to_file_name_{split}.pickle"

    id_to_file = open_pickle(IMAGE_INFO)

    boxes = []
    for result in results:
        image_id = result['image_id']
        c = result['category_id']
        box_score = result['score']
        x, y, w, h = result['bbox']

        box = BoundingBox(
            c,
            x - w // 2,
            y - h // 2,
            x + w // 2,
            y + h // 2,
            box_score
        )

        image = Image(PATH_TO_DIR_UNPROCESSED, id_to_file[image_id])
        h, w = image.image.shape[:2]

        boxes.append(box_to_result_format_exact_category(box.rescale_box(w, h), image_id, c))
    print(boxes)
    to_json(boxes, f"coco_results/resized/{split}_results_score={str(int(score * 100))}.json")


def filter_results_test(split, score):
    results = open_json(f"coco_results/{split}_results_score=30.json")
    file = f"coco_results/{split}_results_score={str(int(score * 100))}.json"
    to_json(filter_results(score, results), file)
    statistics(file, PATH_TO_TEST)


def test_bounding_boxes(split, score, id=None):
    if split == 'test-dev':
        PATH_TO_DIR = PATH_TO_TEST
        PATH_TO_DIR_UNPROCESSED = PATH_TO_TEST_UNPROCESSED
    elif split == 'val':
        PATH_TO_DIR = PATH_TO_VALIDATION
        PATH_TO_DIR_UNPROCESSED = PATH_TO_VALIDATION_UNPROCESSED
    else:
        raise Exception('Invalid split')
    results = open_json(f"coco_results/resized/{split}_results_score={str(int(score * 100))}.json")
    IMAGE_INFO = f"D:\\datasets\\coco\\annotations\\id_to_file_name_{split}.pickle"

    id_to_file = open_pickle(IMAGE_INFO)

    if id is None:
        id = random.choice(list(id_to_file.keys()))

    image = Image(PATH_TO_DIR, id_to_file[id])

    filtered_results = [result for result in results if result['image_id'] == id]
    boxes = []
    for result in filtered_results:
        c = result['category_id']
        score = result['score']
        x, y, w, h = result['bbox']

        boxes.append(BoundingBox(
            c,
            x - w // 2,
            y - h // 2,
            x + w // 2,
            y + h // 2,
            score
        ))

    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.title("Original")
    plt.imshow(image.with_bboxes())

    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.title("Predicted")
    plt.imshow(with_bounding_boxes(image.image, boxes))

    image = Image(PATH_TO_DIR_UNPROCESSED, id_to_file[id])
    h, w = image.image.shape[:2]
    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.imshow(image.with_bboxes())

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.imshow(with_bounding_boxes(image.image, [b.rescale_box(w, h) for b in boxes]))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    split = 'test-dev'
    score = 0.15
    convert_bounding_boxes(split, score)  # 252219
    #test_bounding_boxes(split, score)
