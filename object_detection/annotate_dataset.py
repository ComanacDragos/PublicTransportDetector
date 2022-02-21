# Code adapted from: https://github.com/alitourani/yolo-license-plate-detection
import os

import numpy as np
from cv2 import cv2 as cv

# Initialize the parameters
from matplotlib import pyplot as plt

from generator import DataGenerator
from image import Image, BoundingBox, iou_bbox
from settings import *
from utils import run_task

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     # Width of network's input image
inpHeight = 416  # 608     # Height of network's input image

"""
classesFile = "weights/license_plate_weights/classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
"""

modelConfiguration = "weights/license_plate_weights/darknet-yolov3.cfg"
modelWeights = "weights/license_plate_weights/model.weights"


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    #print(net.getUnconnectedOutLayers())
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    #return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(image: Image, outs):
    frameHeight = image.image.shape[0]
    frameWidth = image.image.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            #if detection[4] > confThreshold:
                #print(detection[4], " - ", scores[classId],
                #      " - th : ", confThreshold)
                #print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        #i = i[0]
        box = boxes[i]
        x_min = box[0]
        y_min = box[1]
        width = box[2]
        height = box[3]
        x_max = x_min + width
        y_max = y_min + height
        new_bbox = BoundingBox(2, x_min, y_min, x_max, y_max, confidences[i])
        max_iou = -1
        for bbox in image.bounding_boxes:
            if new_bbox.c == bbox.c:
                iou = iou_bbox(bbox, new_bbox)
                if iou > max_iou:
                    max_iou = iou
        if max_iou > 0.5:
            print("already exists")
            continue
        print(image.image_name)
        image.bounding_boxes.append(BoundingBox(2, x_min, y_min, x_max, y_max, confidences[i]))


def run_one_image(image: Image, net):
    blob = cv.dnn.blobFromImage(
        image.image, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(image, outs)


def visualize():
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    generator = DataGenerator(PATH_TO_TRAIN, 8, (IMAGE_SIZE, IMAGE_SIZE, 3))
    #generator.on_epoch_end()
    images = [Image(PATH_TO_TRAIN, generator.image_paths[index]) for index in generator.indices[:generator.batch_size]]
    #images = [Image(PATH_TO_TEST, "33881f031110303c.jpg")]
    plt.figure(figsize=(10, 20))
    for i, img in enumerate(images):
        plt.subplot(len(images), 1, i + 1)
        print(f"start {i}'th image")
        #img.bounding_boxes = [bbox for bbox in img.bounding_boxes if bbox.c == 2]
        #run_one_image(img, net)
        print(f"done with {i}'th image\n")
        plt.imshow(img.with_bboxes(3))
    plt.show()


def annotate_worker(image_files, dir, new_dir, net):
    for i, image_file in enumerate(image_files):
        image = Image(dir, image_file)
        run_one_image(image, net)
        image.save_image(new_dir)
        print(f"{i+1}/{len(image_files)}")


def preprocess_dataset():
    dirs = {
        PATH_TO_TRAIN_SIMPLE: PATH_TO_TRAIN,
        PATH_TO_VALIDATION_SIMPLE: PATH_TO_VALIDATION,
        PATH_TO_TEST_SIMPLE: PATH_TO_TEST,
    }
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    for dir, new_dir in dirs.items():
        print(new_dir, " starting...")
        image_files = os.listdir(dir)
        image_files.remove("Label")
        #run_task(image_files, annotate_worker, [dir, new_dir])
        annotate_worker(image_files, dir, new_dir, net)


if __name__ == '__main__':
    print("starting...")
    visualize()
    #preprocess_dataset()