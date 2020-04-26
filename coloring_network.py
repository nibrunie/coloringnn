import sys
import argparse

import os
from os import listdir
from os.path import isfile, join

import random

import numpy as np
import cv2

from tensorflow import keras
from tensorflow.keras import layers

# loading images

# Load an color image in grayscale
# img = cv2.imread('messi5.jpg',0)

def load_samples(dataset_path, resized_dim=(128, 128), size=100):
    color_dataset_path = os.path.join(dataset_path, "color")
    gray_dataset_path = os.path.join(dataset_path, "gray")

    color_images = [f for f in listdir(color_dataset_path) if isfile(join(color_dataset_path, f))]
    gray_images = [f for f in listdir(gray_dataset_path) if isfile(join(gray_dataset_path, f))]

    print("{} color image(s) / {} gray image(s)".format(len(color_images), len(gray_images)))

    sample_array = []

    for img_name in color_images[:size]:
        color_path = os.path.join(color_dataset_path, img_name)
        gray_path = os.path.join(gray_dataset_path, img_name)
        if isfile(color_path) and isfile(gray_path):
            print("found matching input and expected {}".format(img_name))
            color_img = cv2.imread(color_path)
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            color_img = cv2.resize(color_img, resized_dim)
            gray_img = cv2.resize(gray_img, resized_dim)
            cv2.imwrite("gray_img_loaded.png", gray_img)
            sample = color_img, gray_img
            sample_array.append(sample)

    return sample_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='coloring CNN')
    parser.add_argument("--train", type=int, default=None,
                       help="train network")
    parser.add_argument("--model-path", default="colouring_model.model",
                        help="path where the network model is loaded/saved")
    parser.add_argument("--reset-model", action="store_const", default=False,
                        const=True,
                       help="train network")
    parser.add_argument("--eval-on-img", type=str, default=None,
                       help="evaluate trained network on a specific image")
    parser.add_argument("--dataset", type=str, default=None,
                   help="train network")
    parser.add_argument("--dataset-size", type=int, default=100,
                   help="size of the dataset subset to use during training")

    args = parser.parse_args()

    RESIZED_DIM = 128, 128
    GRAY_DIM = RESIZED_DIM + (1,)
    COLOR_DIM = RESIZED_DIM + (3,)

    if args.dataset is None:
        sample_array = []
    else:
        sample_array = load_samples(args.dataset, size=args.dataset_size)
        print("len of sample array: {} elt(s)".format(len(sample_array)))


    if args.reset_model:
        inputs = keras.Input(shape=GRAY_DIM, name='gray_image')
        x = layers.Conv2D(16, 3, activation='relu', name='conv2d_1')(inputs)
        #x = layers.Conv2D(128, 3, activation='relu', name='conv2d_2')(x)
        #x = layers.Conv2D(128, 3, activation='relu', name='conv2d_2')(x)
        x = layers.AveragePooling2D(16, (4,4), name='avg_pool')(x)
        x = layers.Conv2DTranspose(16, 5, name='conv2d_transpose')(x)
        x = layers.UpSampling2D((4, 4), name='up_sampling')(x)
        x = layers.Conv2D(3, 1, activation='relu', name='colored_image')(x)
        input_duplicate = layers.Concatenate(name="input_repeat")([inputs, inputs, inputs])
        #duplicate_reshape = layers.Reshape((128, 128, 3))(input_duplicate)
        outputs = layers.Multiply()([x, input_duplicate])
        outputs = layers.Conv2D(3, 1, activation='relu', name="conv2d_on_forward_inputs")(outputs)
        outputs = layers.Add()([x, outputs])
        outputs = layers.Conv2D(3, 1, activation='relu', name="final_conv2d")(outputs)
        #outputs = layers.Dense(10, name='predictions')(x)


        model = keras.Model(inputs=inputs, outputs=outputs)


        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.RMSprop())


    else:
        model = keras.models.load_model(args.model_path)

    # plotting model
    # TODO/FIXME: error on graphviz/pydot import
    keras.utils.plot_model(model, 'colouring_model.png', show_shapes=True)

    if not args.train is None and not args.dataset is None:
        x_train = np.stack([gray_img.reshape(GRAY_DIM) for _, gray_img in sample_array])
        y_train = np.stack([colored_img for colored_img, _ in sample_array])

        history = model.fit(x_train, y_train,
                            batch_size=64,
                            epochs=args.train,
                            validation_split=0.2)

        # TODO/FIXME using training sample as validation sample
        test_scores = model.evaluate(x_train, y_train, verbose=2)
        print('Test scores:', test_scores)

    keras.models.save_model(model, args.model_path)

    if not args.eval_on_img is None:
        # random_expected, random_input = random.choice(sample_array)
        color_img = cv2.imread(args.eval_on_img)
        color_img = cv2.resize(color_img, (128, 128))
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        model_prediction = model.predict(gray_img.reshape((1,) + GRAY_DIM))
        print("prediction's shape: ", model_prediction.shape)
        predicted_image = model_prediction.reshape(COLOR_DIM)
        print("predicted_image's shape: ", predicted_image.shape)
        cv2.imwrite("random_input.png", gray_img)
        cv2.imwrite("predicted_image.png", predicted_image)
        cv2.imwrite("random_expected.png", color_img)

# result visualisation
#result_viz = cv2.vconcat([random_expected, predicted_image])
# GUI
# window_name = 'image'
# cv2.imshow(window_name, result_viz)

