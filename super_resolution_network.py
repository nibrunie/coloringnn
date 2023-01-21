import sys
import argparse
from datetime import datetime

import os
from os import listdir
from os.path import isfile, join

import random
import math

import numpy as np
import cv2

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from medium_model import generate_super_reso_medium_model

# loading images

def load_dataset(dataset_path, resized_dim=(128, 128), super_size=(256, 256), size=100):
    color_dataset_path = os.path.join(dataset_path, "color")
    super_dataset_path = os.path.join(dataset_path, "super")

    color_images = [f for f in listdir(color_dataset_path) if isfile(join(color_dataset_path, f))]
    super_images = [f for f in listdir(super_dataset_path) if isfile(join(super_dataset_path, f))]

    print("{} color image(s) / {} super image(s)".format(len(color_images), len(super_images)))

    sample_array = []

    for img_name in color_images[:size]:
        color_path = os.path.join(color_dataset_path, img_name)
        super_path = os.path.join(super_dataset_path, img_name)
        if isfile(color_path) and isfile(super_path):
            print("found matching input and expected {}".format(img_name))
            color_img = cv2.imread(color_path)
            super_img = cv2.imread(super_path)
            color_img = cv2.resize(color_img, resized_dim)
            super_img = cv2.resize(super_img, super_size)
            sample = color_img, super_img
            sample_array.append(sample)

    return sample_array

def generate_super_reso_basic_model(INPUT_DIM=(64, 64), SUPER_DIM=(512, 512)):
    """ generate a very basic super-resolution model which takes as input a
        small image and try to extend it to a larger version without loosing details """
    upLevels = int(math.log2(SUPER_DIM[0] / INPUT_DIM[0]))
    print(f"{upLevels} level(s) of up-sampling required")
    inputs = keras.Input(shape=INPUT_DIM + (3,), name='small_image')
    x = layers.Conv2D(16, 3, activation='relu', name='conv2d_1')(inputs)
    x = layers.AveragePooling2D(16, (4,4), name='avg_pool')(x)
    x = layers.Conv2DTranspose(16, 5, name='conv2d_transpose')(x)
    x = layers.UpSampling2D((4, 4), name='up_sampling')(x)
    x = layers.Conv2D(3, 1, activation='relu', name='colored_image')(x)
    for i in range(upLevels):
        x = layers.UpSampling2D((2, 2), name=f'up-sampling-{i}')(x)
    outputs = x
    outputs = layers.Conv2D(3, 1, activation='relu', name="conv2d_on_forward_inputs")(outputs)
    outputs = layers.Add()([x, outputs])
    outputs = layers.Conv2D(3, 1, activation='relu', name="final_conv2d")(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  metrics=["cosine_similarity"],
                  optimizer=keras.optimizers.RMSprop())

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='coloring CNN')
    parser.add_argument("--train", type=int, default=None,
                       help="train network (expect epoch number)")
    parser.add_argument("--model-path", default="super_resolution.model",
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
    parser.add_argument("--dataset-size", type=int, default=100,
                   help="size of the dataset subset to use during training")
    parser.add_argument("--shuffle-dataset", action="store_const", const=True, default=False,
                   help="shuffle dataset before training")
    parser.add_argument("--input-size",
                        type=(lambda s: tuple(map(int, s.split(',')))), default=(128, 128),
                        help="dataset training input dimension")
    parser.add_argument("--super-size",
                        type=(lambda s: tuple(map(int, s.split(',')))), default=(256, 256),
                        help="dataset output dimension (super size)")
    parser.add_argument("--model-type", type=str, choices=["basic", "medium"],
                   default="basic",
                   help="define the type of CNN to use")

    args = parser.parse_args()

    INPUT_DIM = args.input_size
    SUPER_DIM = args.super_size

    if args.dataset is None:
        sample_array = []
    else:
        sample_array = load_dataset(args.dataset, resized_dim=INPUT_DIM, super_size=SUPER_DIM, size=args.dataset_size)
        print("len of sample array: {} elt(s)".format(len(sample_array)))


    if args.reset_model:
        if args.model_type == "basic":
            model = generate_super_reso_basic_model(INPUT_DIM, SUPER_DIM)
        if args.model_type == "medium":
            model = generate_super_reso_medium_model(INPUT_DIM + (3, ), SUPER_DIM + (3, ))
        else:
            raise NotImplementedError

    else:
        model = keras.models.load_model(args.model_path)

    # plotting model
    # TODO/FIXME: error on graphviz/pydot import
    keras.utils.plot_model(model, 'colouring_model.png', show_shapes=True)


    if args.shuffle_dataset:
        random.shuffle(sample_array)

    # preparing training samples (outside of train block to be able to
    # inject them into the summary)
    if not args.train is None and not args.dataset is None:
        x_train = np.stack([color_img for color_img, _ in sample_array])
        y_train = np.stack([super_img for _, super_img in sample_array])

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
                            batch_size=16,
                            epochs=args.train,
                            validation_split=0.05,
                            # verbose=0,
                            callbacks=[tensorboard_callback, valid_state_callback])

        # TODO/FIXME using training sample as validation sample
        test_scores = model.evaluate(x_train, y_train, verbose=2)
        print('Test scores: ', test_scores)


    if args.print_model_summary:
        print(model.summary())

    keras.models.save_model(model, args.model_path)

    if not args.eval_on_img is None:
        # random_expected, random_input = random.choice(sample_array)
        color_img = cv2.imread(args.eval_on_img)
        input_img = cv2.resize(color_img, INPUT_DIM)
        super_img = cv2.resize(color_img, SUPER_DIM)
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # reshape_input = gray_img.reshape((3,) + INPUT_DIM)
        reshape_input = input_img.reshape((1,) + INPUT_DIM + (3,))
        model_prediction = model.predict(reshape_input)
        print("prediction's shape: ", model_prediction.shape, model_prediction.dtype)
        predicted_image = model_prediction.reshape(SUPER_DIM + (3,))
        print("color_img's shape: ", color_img.shape, color_img.dtype)
        print("predicted_image's shape: ", predicted_image.shape, predicted_image.dtype)
        print("predicted_image's range", np.amax(predicted_image), np.amin(predicted_image))
        predicted_image_u8 = predicted_image.astype('uint8')
        print("predicted_image_u8's range", np.amax(predicted_image_u8), np.amin(predicted_image_u8))
        print("predicted_image's shape: ", predicted_image.shape)



        cv2.imwrite("input_small-image.png", input_img)
        cv2.imwrite("predicted_super-image.png", predicted_image)

        # # evaluation logging
        # logdir_eval = "logs/eval_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # file_writer_eval = tf.summary.create_file_writer(logdir_eval)

        # ext_input = np.repeat(gray_img, 3).reshape(COLOR_DIM)
        # print("ext_input's shape: ", ext_input.shape, ext_input.dtype)

        # img_stack = np.stack([ext_input, color_img[...,::-1], predicted_image.astype('uint8')[...,::-1]])
        # print("img_stack's properties: ", img_stack.shape, img_stack.dtype, np.amax(img_stack), np.amin(img_stack))
        # img_stack = img_stack.astype('float32') / 255.0
        # print("img_stack's properties: ", img_stack.shape, img_stack.dtype, np.amax(img_stack), np.amin(img_stack))
        # with file_writer_eval.as_default():
        #     tf.summary.image("evaluation results", img_stack, step=0)

