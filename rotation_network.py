import sys
import argparse
from datetime import datetime

import os
from os import listdir
from os.path import isfile, join

import random

import numpy as np
import cv2

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf


# loading images
def load_samples(dataset_path, resized_dim=(128, 128), size=100):
    dataset_path = os.path.join(dataset_path)

    images = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

    print(f"{len(images)} image(s)")

    sample_array = []

    rotateFuncs = {
        0: (lambda img: img),
        90: (lambda img: cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)),
        180: (lambda img: cv2.rotate(img, rotateCode=cv2.ROTATE_180)),
        270: (lambda img: cv2.rotate(img, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)),
    }

    for img_name in images[:size]:
        img_path = os.path.join(dataset_path, img_name)
        if isfile(img_path):
            # print("found matching input and expected {}".format(img_name))
            color_img = cv2.imread(img_path)
            color_img = cv2.resize(color_img, resized_dim)
            angle = random.choices([0, 90, 180, 270], k=1)[0]
            color_img = rotateFuncs[angle](color_img)
            sample = color_img, angle
            sample_array.append(sample)

    return sample_array

def generate_basic_model(IMG_DIM=(128, 128, 3)):
    """ generate a very basic coloring model which takes as input a
        gray image and try to colorize it """
    model = Sequential([
        layers.Rescaling(1./255, input_shape=IMG_DIM),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(4)
    ])
    # inputs = keras.Input(shape=IMG_DIM, name='input_image')
    # x = layers.Conv2D(16, (3, 3), input_shape=IMG_DIM, activation='relu', name='conv2d_1')(inputs)
    # x = layers.AveragePooling2D(16, (4,4), name='avg_pool')(x)
    # x = layers.Conv2D(16, (3, 3), name='conv2d_transpose')(x)
    # x = layers.MaxPool2D((4, 4), name='max_pooling')(x)
    # x = layers.Conv2D(3, (4, 4), activation='relu', name='colored_image_lvl1')(x)
    # x = layers.Conv2D(1, (3, 3), activation='relu', name='colored_image_lvl2')(x)
    # # normalized_prob = layers.Normalization()(layers.Dense(4, action='relu')(x))
    # normalized_prob = layers.Dense(4)(x)
    # outputs = tf.keras.layers.Flatten()(normalized_prob)
    # model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"],
                  optimizer='adam')

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='coloring CNN')
    parser.add_argument("--train", type=int, default=None,
                       help="train network (expect epoch number)")
    parser.add_argument("--model-path", default="colouring_model.model",
                        help="path where the network model is loaded/saved")
    parser.add_argument("--reset-model", action="store_const", default=False,
                        const=True,
                       help="reset the network model")
    parser.add_argument("--print-model-summary", action="store_const", default=False,
                        const=True,
                       help="print model summary")
    parser.add_argument("--eval-on-img", type=str, default=None,
                       help="evaluate trained network on a specific image")
    parser.add_argument("--dataset", type=str, default=None,
                   help="train network")
    parser.add_argument("--shuffle-dataset", action="store_const", const=True, default=False,
                   help="shuffle dataset before training")
    parser.add_argument("--dataset-size", type=int, default=100,
                   help="size of the dataset subset to use during training")
    parser.add_argument("--model-type", type=str, choices=["basic", "medium"],
                   default="basic",
                   help="define the type of CNN to use")

    args = parser.parse_args()

    RESIZED_DIM = 128, 128
    COLOR_DIM = RESIZED_DIM + (3,)

    if args.dataset is None:
        sample_array = []
    else:
        sample_array = load_samples(args.dataset, size=args.dataset_size)
        print("len of sample array: {} elt(s)".format(len(sample_array)))


    if args.reset_model:
        if args.model_type == "basic":
            model = generate_basic_model(COLOR_DIM)
        else:
            raise NotImplementedError

    else:
        model = keras.models.load_model(args.model_path)

    # plotting model
    # TODO/FIXME: error on graphviz/pydot import
    keras.utils.plot_model(model, 'rotation_model.png', show_shapes=True)


    if args.shuffle_dataset:
        random.shuffle(sample_array)

    angleMap = {
        0:   np.array([1, 0, 0, 0]),
        90:  np.array([0, 1, 0, 0]),
        180: np.array([0, 0, 1, 0]),
        270: np.array([0, 0, 0, 1]),
    }
    angleIndexMap = {a: i for (i, a) in enumerate([0, 90, 180, 270])}

    # preparing training samples (outside of train block to be able to
    # inject them into the summary)
    if not args.train is None and not args.dataset is None:
        x_train = np.stack([img for img, _ in sample_array])
        y_train = np.stack([angleIndexMap[angle] for _, angle in sample_array])
        # y_train = np.asarray(y_train) #.astype('float32').reshape((-1, 4))

        print(f"y_train's shape: {y_train.shape}")

        # logging training
        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        # Sets up a timestamped log directory.
        logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # Creates a file writer for the log directory.
        file_writer = tf.summary.create_file_writer(logdir)
        with file_writer.as_default():
            tf.summary.image(" 25 traiing samples", x_train[0:25], max_outputs=25, step=0)

        # validation logging
        logdir_valid = "logs/valid_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer_valid = tf.summary.create_file_writer(logdir_valid)

        # logging validation state
        def log_valid_state(epoch, logs):
            epoch_predictions = model.predict(x_train[0:25])
            # BGR to RGB
            epoch_predictions = epoch_predictions[...,::-1]
            # convert to float and scaling to [0, 1.0)
            epoch_predictions = epoch_predictions.astype('float32') / 255.0

            with file_writer_valid.as_default():
                tf.summary.image("validation results", epoch_predictions, step=epoch)

        valid_state_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_valid_state)

        history = model.fit(x_train, y_train,
                            batch_size=64,
                            epochs=args.train,
                            validation_split=0.05,
                            # verbose=0,
                            #callbacks=[tensorboard_callback, valid_state_callback]
        )

        # TODO/FIXME using training sample as validation sample
        test_scores = model.evaluate(x_train, y_train, verbose=2)
        print('Test scores: ', test_scores)


    if args.print_model_summary:
        print(model.summary())

    keras.models.save_model(model, args.model_path)

    if not args.eval_on_img is None:
        # random_expected, random_input = random.choice(sample_array)
        color_img = cv2.imread(args.eval_on_img)
        color_img = cv2.resize(color_img, (128, 128))

        # image must be layed down in a batch-1 4D array
        reshape_input = color_img.reshape((1,) + COLOR_DIM)
        model_prediction = model.predict(reshape_input)
        print("prediction's shape: ", model_prediction.shape, model_prediction.dtype)
        print(f"model prediction: {model_prediction}")
        predicted_angle = [0, 90, 180, 270][np.argmax(model_prediction)]
        print(f"predicated angle: {predicted_angle}")