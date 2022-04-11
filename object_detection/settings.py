import logging
from utils_file import open_pickle

ENHANCE = True
UNDER_SAMPLE = True
USE_COCO = True

if not USE_COCO:
    if not ENHANCE:
        if not UNDER_SAMPLE:
            # PATH_TO_TRAIN = "E:\\Universitate\\Licenta\\OIDv4\\OID\\PublicTransport\\train\\Bus_Car_Vehicle registration plate"
            PATH_TO_TRAIN = "C:\\Users\\Dragos\\datasets\\PublicTransportProcessed\\train"
            PATH_TO_TRAIN_UNPROCESSED = "C:\\Users\\Dragos\\datasets\\PublicTransport\\train\\Bus_Car_Vehicle registration plate"

            # PATH_TO_VALIDATION = "E:\\Universitate\\Licenta\\OIDv4\\OID\\PublicTransport\\validation\\Bus_Car_Vehicle registration plate" # hdd
            PATH_TO_VALIDATION = "C:\\Users\\Dragos\\datasets\\PublicTransportProcessed\\validation"  # ssd
            PATH_TO_VALIDATION_UNPROCESSED = "C:\\Users\\Dragos\\datasets\\PublicTransport\\validation\\Bus_Car_Vehicle registration plate"  # ssd

            # PATH_TO_TEST = "E:\\Universitate\\Licenta\\OIDv4\\OID\\PublicTransport\\test\\Bus_Car_Vehicle registration plate"
            PATH_TO_TEST = "C:\\Users\\Dragos\\datasets\\PublicTransportProcessed\\test"
            PATH_TO_TEST_UNPROCESSED = "C:\\Users\\Dragos\\datasets\\PublicTransport\\test\\Bus_Car_Vehicle registration plate"
        else:
            PATH_TO_TRAIN = "C:\\Users\\Dragos\\datasets\\OID\\OID\\PublicTransportFilteredProcessed\\train"
            PATH_TO_TRAIN_UNPROCESSED = "C:\\Users\\Dragos\\datasets\\OID\\OID\\PublicTransportFiltered\\train\\Bus_Car_Vehicle registration plate"

            PATH_TO_VALIDATION = "C:\\Users\\Dragos\\datasets\\OID\\OID\\PublicTransportFilteredProcessed\\validation"  # ssd
            PATH_TO_VALIDATION_UNPROCESSED = "C:\\Users\\Dragos\\datasets\\OID\\OID\\PublicTransportFiltered\\validation\\Bus_Car_Vehicle registration plate"  # ssd

            PATH_TO_TEST_FILTERED = "C:\\Users\\Dragos\\datasets\\OID\\OID\\PublicTransportFilteredProcessed\\test_filtered"
            PATH_TO_TEST = PATH_TO_TEST_FILTERED# "C:\\Users\\Dragos\\datasets\\OID\\OID\\PublicTransportFilteredProcessed\\test"
            PATH_TO_TEST_UNPROCESSED = "C:\\Users\\Dragos\\datasets\\OID\\OID\\PublicTransportFiltered\\test\\Bus_Car_Vehicle registration plate"
    else:
        PREFIX = "C:\\Users\\Dragos\\datasets\\OID\\OID"
        PATH_TO_TRAIN_SIMPLE = f"{PREFIX}\\PublicTransportFilteredProcessed\\train"
        PATH_TO_VALIDATION_SIMPLE = f"{PREFIX}\\PublicTransportFilteredProcessed\\validation"
        PATH_TO_TEST_SIMPLE = f"{PREFIX}\\PublicTransportFilteredProcessed\\test_filtered"

        PATH_TO_TRAIN = f"{PREFIX}\\PublicTransportEnhanced\\train"
        PATH_TO_VALIDATION = f"{PREFIX}\\PublicTransportEnhanced\\validation"
        PATH_TO_TEST = f"{PREFIX}\\PublicTransportEnhanced\\test"
else:
    UNPROCESSED_PREFIX = "D:\\datasets\\coco"
    PROCESSED_PREFIX = "C:\\Users\\Dragos\\datasets\\coco" #"D:\\datasets\\coco\\coco_resized"

    PATH_TO_TRAIN = f"{PROCESSED_PREFIX}\\train"
    PATH_TO_TRAIN_UNPROCESSED = f"{UNPROCESSED_PREFIX}\\train2017"

    PATH_TO_VALIDATION = f"{PROCESSED_PREFIX}\\val"
    PATH_TO_VALIDATION_UNPROCESSED = f"{UNPROCESSED_PREFIX}\\val2017"

    PATH_TO_TEST = f"{PROCESSED_PREFIX}\\test"
    PATH_TO_TEST_UNPROCESSED = f"{UNPROCESSED_PREFIX}\\test2017"


ENCODE_LABEL = {}
if USE_COCO:
    CATEGORIES_PATH = "D:\\datasets\\coco\\annotations\\categories.pickle"
    categories = open_pickle(CATEGORIES_PATH)

    for i, label in enumerate(categories.values()):
        ENCODE_LABEL[label] = i

else:
    ENCODE_LABEL = {
        "Bus": 0,
        "Car": 1,
        "Vehicle registration plate": 2
    }

DECODE_LABEL = {v: k for k, v in ENCODE_LABEL.items()}

# Data processing
IMAGE_SIZE = 416
GRID_SIZE = 13

if USE_COCO:
    MAX_BOXES_PER_IMAGES = 91 * 4
    NO_ANCHORS = 5
    ANCHORS_PATH = f"data/coco_anchors_5.pickle"
else:
    MAX_BOXES_PER_IMAGES = 111

    NO_ANCHORS = 3
    ANCHORS_PATH = f"data/anchors_{NO_ANCHORS}.pickle"

BATCH_SIZE = 32


logging.basicConfig(level=logging.INFO)
