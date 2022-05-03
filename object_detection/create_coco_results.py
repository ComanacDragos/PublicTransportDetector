import os
from utils_file import *
from settings import *
from inference import *

IMAGE_INFO = "D:\\datasets\\coco\\annotations\\id_to_file_name_val.pickle"
PATH_TO_DIR = PATH_TO_VALIDATION


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
    id_to_file = open_pickle(IMAGE_INFO)

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
        bboxes = image_to_bounding_boxes(model, f"{PATH_TO_DIR}/{file}")
        image_results = [box_to_result_format(box, id, category_to_id) for box in bboxes]
        results += image_results
    print(results)
    to_json(results, "coco_results/val_results.json")
