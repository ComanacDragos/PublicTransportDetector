import os

from utils_file import to_pickle, open_pickle, open_json


def extract_image_file_names(images_data):
    """
    Extracts the ids and the file names

    :param images_data: data about the images
    :return: id to file name dictionary
    """
    id_to_file_name = {}
    for image in images_data:
        id_to_file_name[image['id']] = image['file_name']
    return id_to_file_name


def extract_bounding_boxes(annotations, categories, id_to_file_name):
    """
    Extracts for each image the bounding boxes

    :param annotations: COCO annotations
    :param categories: COCO id to category
    :param id_to_file_name: dictionary mapping the id of an image to it's file name
    :return: dictionary mapping each file to it's corresponding bounding boxes
    """
    file_to_boxes = {}
    for annotation in annotations:
        file_name = id_to_file_name[annotation['image_id']]
        category = categories[annotation['category_id']]
        x, y, width, height = annotation['bbox']

        if width == 0 or height == 0:
            print("Invalid box in ", file_name)
            continue

        bbox = {
            'label': category,
            'min_x': x,
            'min_y': y,
            'max_x': x+width,
            'max_y': y+height
        }

        if file_name in file_to_boxes:
            file_to_boxes[file_name].append(bbox)
        else:
            file_to_boxes[file_name] = [bbox]
    return file_to_boxes


def prepare_annotations(split):
    annotation_data = open_json(f"D:\\datasets\\coco\\annotations_trainval2017\\{split}.json")
    print("Done loading...")
    annotations_root = "D:\\datasets\\coco\\annotations"
    categories = open_pickle(f"{annotations_root}\\categories.pickle")
    id_to_file_name = open_pickle(f"{annotations_root}\\id_to_file_name_{split}.pickle")

    print(len(id_to_file_name))

    annotations = annotation_data['annotations']

    file_to_boxes = extract_bounding_boxes(annotations, categories, id_to_file_name)

    file_path = f"{annotations_root}\\file_to_boxes_{split}.pickle"
    to_pickle(file_to_boxes, file_path)
    print(len(open_pickle(file_path)))
    """
    to_pickle(extract_image_file_names(annotation_data['images']), f"{annotations_root}\\id_to_file_name_train.pickle")

    print(len(open_pickle(f"{annotations_root}\\id_to_file_name_train.pickle")))
    """


def prepare_labels(split):
    file_to_boxes = open_pickle(f"D:\\datasets\\coco\\annotations\\file_to_boxes_{split}.pickle")
    dataset_root = f"D:\\datasets\\coco\\{split}2017"
    paths_to_images = os.listdir(dataset_root)
    paths_to_images.remove('Label')
    print(len(paths_to_images))
    print(paths_to_images[:10])

    for path in paths_to_images:
        root = path.split('.')[0]
        with open(f"{dataset_root}\\Label\\{root}.txt", 'w') as f:
            if path in file_to_boxes:
                for box in file_to_boxes[path]:
                    f.write(f"{box['label']} {box['min_x']} {box['min_y']} {box['max_x']} {box['max_y']}\n")


def create_test_set_label_files():
    dataset_root = f"D:\\datasets\\coco\\test2017"
    paths_to_images = os.listdir(dataset_root)
    paths_to_images.remove('Label')
    print(len(paths_to_images))
    print(paths_to_images[:10])

    for path in paths_to_images:
        root = path.split('.')[0]
        with open(f"{dataset_root}\\Label\\{root}.txt", 'w'):
            pass

if __name__ == '__main__':
    print("Starting...")
    #prepare_annotations('val')
    #prepare_labels('val')
    #create_test_set_label_files()
    #print(len(os.listdir("D:\\datasets\\coco\\train2017")))

"""
    annotation_data = open_json(f"D:\\datasets\\coco\\image_infoannotations\\image_info_test-dev2017.json")

    to_pickle(extract_image_file_names(annotation_data['images']), "D:\\datasets\\coco\\annotations\\id_to_file_name_test-dev.pickle")

    print(open_pickle("D:\\datasets\\coco\\annotations\\id_to_file_name_test.pickle"))
"""
