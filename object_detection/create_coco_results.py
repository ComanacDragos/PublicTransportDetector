import os
from utils_file import *
from settings import *
from inference import *


def box_to_result_format(box: BoundingBox, image_id, category_to_id):
    x, y = box.center()
    w, h = box.width_height()

    return {
        "image_id": image_id,
        "category_id": category_to_id[DECODE_LABEL[box.c]],
        "bbox": [x, y, w, h],
        "score": round(float(box.score), 4)
    }


if __name__ == '__main__':
    IMAGE_INFO = "D:\\datasets\\coco\\annotations\\id_to_file_name_test-dev.pickle"
    PATH_TO_DIR = PATH_TO_TEST

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
        bboxes = image_to_bounding_boxes(model, f"{PATH_TO_DIR}/{file}", score_threshold=0.15)
        image_results = [box_to_result_format(box, id, category_to_id) for box in bboxes]
        results += image_results
    print(results)
    to_json(results, "coco_results/test-dev_results_score=015.json")
