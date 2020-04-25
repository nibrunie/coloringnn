from tensorflow import keras
from tensorflow.keras import layers

import os
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2


# loading images

# Load an color image in grayscale
# img = cv2.imread('messi5.jpg',0)

def load_samples(resized_dim=(128, 128)):
    dataset_path = os.path.join("dataset")
    color_dataset_path = os.path.join(dataset_path, "color")
    gray_dataset_path = os.path.join(dataset_path, "gray")

    color_images = [f for f in listdir(color_dataset_path) if isfile(join(color_dataset_path, f))]
    gray_images = [f for f in listdir(gray_dataset_path) if isfile(join(gray_dataset_path, f))]

    print("{} color image(s) / {} gray image(s)".format(len(color_images), len(gray_images)))

    sample_array = []

    for img_name in color_images:
        color_path = os.path.join(color_dataset_path, img_name)
        gray_path = os.path.join(gray_dataset_path, img_name)
        if isfile(color_path) and isfile(gray_path):
            print("found matching input and expected {}".format(img_name))
            color_img = cv2.imread(color_path)
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            sample = color_img.resize(resized_dim + (3,)), gray_img.resize(resized_dim + (1,))
            sample_array.append(sample)

    return sample_array


sample_array = load_samples()
print("len of sample array: {} elt(s)".format(len(sample_array)))

RESIZED_DIM = 128, 128
GRAY_DIM = RESIZED_DIM + (1,)
COLOR_DIM = RESIZED_DIM + (3,)

inputs = keras.Input(shape=GRAY_DIM, name='gray_image')
x = layers.Conv2D(16, 3, activation='relu', name='conv2d_1')(inputs)
#x = layers.Conv2D(128, 3, activation='relu', name='conv2d_2')(x)
#x = layers.Conv2D(128, 3, activation='relu', name='conv2d_2')(x)
x = layers.AveragePooling2D(16, (3,3), name='avg_pool')(x)
x = layers.UpSampling2D((3, 3), name='up_sampling')(x)
outputs = layers.Conv2D(3, 1, activation='relu', name='colored_image')(inputs)
#outputs = layers.Dense(10, name='predictions')(x)


#model = keras.Model(inputs=inputs, outputs=outputs)

