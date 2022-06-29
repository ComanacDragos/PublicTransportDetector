import random
import time

import cv2.cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

from inference import draw_images, inference
from train import *


class VideoGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_file, batch_size=BATCH_SIZE, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
        self.input_shape = input_shape
        self.batch_size = batch_size

        self.video_file = video_file
        self.video_length = cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_COUNT)

        self.anchors = process_anchors(ANCHORS_PATH)
        self.indices = np.arange(self.video_length)

        self.frames = []
        video = cv2.VideoCapture(self.video_file)
        ok = True
        while ok:
            ok, frame = video.read()
            if ok:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
                self.frames.append(frame)
        video.release()

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.ceil(self.video_length / self.batch_size))

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        #video = cv2.VideoCapture(self.video_file)
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        """
        batch = []
    
        for i in batch_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, frame = video.read()
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
            batch.append(frame)
        video.release()
        """

        return np.asarray([self.frames[int(i)] for i in batch_indices])


def test(path_to_model, score_threshold=0.2, iou_threshold=0.3, batch_size=BATCH_SIZE, batch=None, _generator=None):
    if _generator is None:
        generator = DataGenerator(PATH_TO_TEST, shuffle=False, batch_size=batch_size)
    else:
        generator = _generator
    print('#Batches:', len(generator))
    model, true_boxes = build_model()
    if "fine_tuned" in path_to_model:
        model.trainable = True
    model.load_weights(path_to_model)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=YoloLoss(anchors=generator.anchors, true_boxes=true_boxes, enable_logs=True))

    if batch is None:
        batch = random.randint(0, len(generator))
    print(f"batch {batch}")
    images = generator[batch]

    start = time.perf_counter()
    scores, boxes, classes, valid_detections = inference(model, images, score_threshold=score_threshold, iou_threshold=iou_threshold)
    scores, boxes, classes, valid_detections = K.get_value(scores), \
                                               K.get_value(boxes), \
                                               K.get_value(classes), \
                                               K.get_value(valid_detections)

    print("time to run: ", time.perf_counter() - start)
    print("Shapes")
    print(f"Scores: {scores.shape}")
    print(f"Boxes: {boxes.shape}")
    print(f"Classes: {classes.shape}")
    print(f"Valid detections: {valid_detections.shape}")

    pred_images_with_boxes = draw_images(images, scores, boxes, classes, valid_detections)

    fig, axs = plt.subplots(BATCH_SIZE, 2, figsize=(20, 40))
    for i in range(0, BATCH_SIZE):
        axs[i][0].imshow(images[i])
        axs[i][1].imshow(pred_images_with_boxes[i])
        axs[i][0].axis("off")
        axs[i][1].axis("off")

    plt.tight_layout()
    #plt.savefig(f"documentation/results/final_results_model_v42/{batch}.jpeg")
    plt.show()

    #img_array = draw_images([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in images], scores, boxes, classes, valid_detections)
    img_array = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in pred_images_with_boxes]

    size = (IMAGE_SIZE, IMAGE_SIZE)
    out = cv2.VideoWriter('video/test.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def detect_video(path_to_model, generator, output_video_name, score_threshold=0.2, iou_threshold=0.3):
    model, true_boxes = build_model()
    if "fine_tuned" in path_to_model:
        model.trainable = True
    model.load_weights(path_to_model)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=YoloLoss(anchors=generator.anchors, true_boxes=true_boxes, enable_logs=True))

    all_images = []
    for batch in range(len(generator)):
        start = time.perf_counter()
        images = generator[batch]
        load_batch_time = time.perf_counter() - start
        start = time.perf_counter()
        scores, boxes, classes, valid_detections = inference(model, images, score_threshold=score_threshold, iou_threshold=iou_threshold)
        scores, boxes, classes, valid_detections = K.get_value(scores), \
                                                   K.get_value(boxes), \
                                                   K.get_value(classes), \
                                                   K.get_value(valid_detections)

        print(f"{batch+1}/{len(generator)} load: {load_batch_time} detect: {time.perf_counter() - start}")
        all_images += draw_images(images, scores, boxes, classes, valid_detections)
    img_array = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in all_images]

    size = (IMAGE_SIZE, IMAGE_SIZE)
    out = cv2.VideoWriter(f'video/{output_video_name}.mp4',
                          cv2.VideoWriter_fourcc(*'XVID'),
                          15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    ROOT = "C:\\Users\\Dragos\\datasets"
    PATH_TO_MODEL = "weights/model_v42_fine_tuned.h5"
    start = time.perf_counter()
    generator = VideoGenerator(f"{ROOT}/video/2.mp4", batch_size=32)
    print(f"Generator load time: {time.perf_counter()-start}")
    #test(path_to_model=PATH_TO_MODEL, _generator=generator)
    detect_video(PATH_TO_MODEL, generator, "2_50_detected", score_threshold=0.5)
