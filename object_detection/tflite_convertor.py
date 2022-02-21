import numpy as np
import tensorflow as tf

from data_augmentation import RandomColorAugmentation, Cutout
from train import build_model


def build_model_for_inference_only(model_name):
    model: tf.keras.Model = tf.keras.models.load_model(f"weights/{model_name}.h5", custom_objects={
        'RandomColorAugmentation': RandomColorAugmentation,
        'Cutout': Cutout
    }, compile=False)
    pruned_model: tf.keras.Model = build_model(inference_only=True)
    for layer in pruned_model.layers:

        if "Div" in layer.name or "Sub" in layer.name:
            continue
        layer.set_weights(model.get_layer(layer.name).get_weights())
    return pruned_model


def convert_model(model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(build_model_for_inference_only("model_v26"))
    tflite_model = converter.convert()

    # Save the model.
    with open(f"tflite/{model_name}.tflite", 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    model_name = "model_v26"
    convert_model(model_name)
