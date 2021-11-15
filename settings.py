# Dataset related constants
#PATH_TO_TRAIN = "E:\\Universitate\\Licenta\\OIDv4\\OID\\PublicTransport\\train\\Bus_Car_Vehicle registration plate"
PATH_TO_TRAIN = "C:\\Users\\Dragos\\datasets\\PublicTransport\\train\\Bus_Car_Vehicle registration plate"

#PATH_TO_VALIDATION = "E:\\Universitate\\Licenta\\OIDv4\\OID\\PublicTransport\\validation\\Bus_Car_Vehicle registration plate" # hdd
PATH_TO_VALIDATION = "C:\\Users\\Dragos\\datasets\\PublicTransport\\validation\\Bus_Car_Vehicle registration plate" # ssd

#PATH_TO_TEST = "E:\\Universitate\\Licenta\\OIDv4\\OID\\PublicTransport\\test\\Bus_Car_Vehicle registration plate"
PATH_TO_TEST = "C:\\Users\\Dragos\\datasets\\PublicTransport\\test\\Bus_Car_Vehicle registration plate"


ENCODE_LABEL = {
    "Bus": 0,
    "Car": 1,
    "Vehicle registration plate": 2
}

# Data processing
IMAGE_SIZE = 416

ANCHORS = 5