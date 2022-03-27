import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

from data_augmentation import RandomColorAugmentation, Cutout
from generator import process_anchors, DataGenerator
from inference import output_processor, draw_images, process_ground_truth
from settings import *
from train import build_model


def build_model_for_tflite(model_name):
    """
    One input:
      image: a float32 tensor of shape[1, height, width, 3] containing the
      *normalized* input image.
      NOTE: See the `preprocess` function defined in the feature extractor class
      in the object_detection/models directory.

    Four Outputs:
      detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box locations
      detection_classes: a float32 tensor of shape [1, num_boxes] with class indices
      detection_scores: a float32 tensor of shape [1, num_boxes] with class scores
      num_boxes: a float32 tensor of size 1 containing the number of detected boxes

    outputMap.put(0, outputScores);
    outputMap.put(1, outputLocations);
    outputMap.put(2, numDetections);
    outputMap.put(3, outputClasses);

    0 (1, no_boxes, 4)
    1 (1, no_boxes)
    2 (1, n0_boxes)
    3 (1,)
    """
    model: tf.keras.Model = tf.keras.models.load_model(f"weights/{model_name}.h5", custom_objects={
        'RandomColorAugmentation': RandomColorAugmentation,
        'Cutout': Cutout
    }, compile=False)
    pruned_model: tf.keras.Model = build_model(inference_only=True)
    for layer in pruned_model.layers:
        if "Div" in layer.name or "Sub" in layer.name:
            continue
        layer.set_weights(model.get_layer(layer.name).get_weights())
    anchors = process_anchors(ANCHORS_PATH)
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = pruned_model(inputs)

    scores, boxes, classes = output_processor(x, anchors)
    boxes = tf.reshape(boxes, (-1, GRID_SIZE * GRID_SIZE * len(anchors), 4)) / IMAGE_SIZE
    classes = tf.reshape(classes, (-1, GRID_SIZE * GRID_SIZE * len(anchors)))
    classes = tf.cast(classes, tf.float32)

    scores = tf.reshape(scores, (-1, GRID_SIZE * GRID_SIZE * len(anchors)))

    hack = tf.keras.Model(inputs=inputs, outputs=[boxes, classes, scores], name="hack_inference_yolo")

    hack_boxes, hack_classes, hack_scores = hack(inputs)
    valid_detections = tf.shape(hack_scores)[1:2]
    valid_detections = tf.cast(valid_detections, tf.float32)

    return tf.keras.Model(inputs=inputs, outputs=[hack_boxes, hack_classes, hack_scores, valid_detections],
                          name="inference_yolo")


def representative_data_set_generator(mean=127.5, norm=127.5):
    generator = DataGenerator(PATH_TO_TRAIN, shuffle=False, batch_size=1)
    for i in range(len(generator)):
        (img, _), _ = generator[i]
        img = (img - mean) / norm
        yield img


def convert_model(model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(build_model_for_tflite(model_name))
    #converter.experimental_new_converter = True

    #converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.representative_dataset = representative_data_set_generator
  
    # converter.allow_custom_ops = True

    tflite_model = converter.convert()
    _MODEL_PATH = f"tflite/{model_name}.tflite"

    with open(_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    ObjectDetectorWriter = object_detector.MetadataWriter
    _LABEL_FILE = "data/labels.txt"
    _SAVE_TO_PATH = _MODEL_PATH

    _INPUT_NORM_MEAN = 127.5
    _INPUT_NORM_STD = 127.5

    # Create the metadata writer.
    writer = ObjectDetectorWriter.create_for_inference(
        writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [_LABEL_FILE])

    # Verify the metadata generated by metadata writer.
    print(writer.get_metadata_json())

    # Populate the metadata into the model.
    writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)


def analyze(model_name):
    interpreter = tf.lite.Interpreter(f"tflite/{model_name}.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    for i in range(len(output_details)):
        output_data = interpreter.get_tensor(output_details[i]['index'])
        print(i, output_data.shape)


if __name__ == '__main__':
    model_name = "model_v28"
    try:
        convert_model(model_name)
        analyze(model_name)
        # build_model_for_inference_only(model_name).summary()
    except Exception as e:
        with open("tflite/error.txt", "w") as f:
            f.write(str(e))
        raise e
    # model = build_model_for_inference_only(model_name)
    # model.summary()
    # print(model.output_shape)
