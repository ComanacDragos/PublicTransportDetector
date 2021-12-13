import tensorflow as tf
from generator import *
from train import build_simple_model
from generator import process_anchors


def output_processor(anchors_path="data/anchors.pickle"):
    anchors = process_anchors(anchors_path)


def inference(model, input):
    dummy_array = np.zeros((1, 1, 1, 1, MAX_BOXES_PER_IMAGES, 4))
    y_pred = model.predict([images, dummy_array])


def test():
    generator = DataGenerator(PATH_TO_TRAIN, 8)
    model, _ = build_simple_model()
    model.load_weights("weights/model.h5")
    # model = tf.keras.models.load_model("weights/model.h5", compile=False)
    ground_truth, y_true = generator[0]
    images, true_boxes = ground_truth[0], ground_truth[1]

    inference(model, images)
    return
    fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 2]})
    for i in range(0, 1):
        original_boxes = interpret_output(y_true[i - 1], generator.anchors)
        predicted_boxes = interpret_output(y_pred[i - 1], generator.anchors)

        print(f"{i} th image:")
        for j in original_boxes:
            print(j)
        print("Predicted:")
        for j in predicted_boxes:
            print(j)

        print()
        original_image = with_bounding_boxes(images[i - 1], original_boxes, 3)
        predicted_image = with_bounding_boxes(images[i - 1], predicted_boxes, 3)

        # plt.subplot(6, 2, 2 * i - 1)
        axs[i][0].imshow(original_image)
        # axs[i][0].title('Original')

        # plt.subplot(6, 2, 2 * i)
        axs[i][1].imshow(predicted_image)
        # axs[i][1].title('Predicted')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()
