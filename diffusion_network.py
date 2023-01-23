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

import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# developped from https://keras.io/examples/generative/ddim/

# loading images
def load_samples(dataset_path, resized_dim=(128, 128), size=100):
    dataset_path = os.path.join(dataset_path)

    images = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

    print(f"{len(images)} image(s)")

    sample_array = []

    for img_name in images[:size]:
        img_path = os.path.join(dataset_path, img_name)
        if isfile(img_path):
            # print("found matching input and expected {}".format(img_name))
            color_img = cv2.imread(img_path)
            color_img = cv2.resize(color_img, resized_dim)
            sample = color_img
            sample_array.append(sample)

    return sample_array

# Hyperparameters
# data
dataset_repetitions = 5
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

class HyperParameters:
    # optimization
    batch_size = 64
    ema = 0.999
    learning_rate = 1e-3
    weight_decay = 1e-4

def preprocess_image(data, image_size):
    # center crop image
    height = tf.shape(data)[0]
    width = tf.shape(data)[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data,
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=image_size, antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)

class KID(keras.metrics.Metric):
    def __init__(self, name, image_size, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.image_size = image_size
        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID("kid", self.image_size)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, self.image_size, self.image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(HyperParameters.batch_size, self.image_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(HyperParameters.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(HyperParameters.ema * ema_weight + (1 - HyperParameters.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(HyperParameters.batch_size, self.image_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(HyperParameters.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=HyperParameters.batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6, saveFile=None):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )
        if saveFile:
            for i, img in enumerate(generated_images):
                predicted_image = img.reshape((self.image_size, self.image_size) + (3,))
                cv2.imwrite(f"{saveFile}-{i}.png", (predicted_image.numpy() * 256).astype("uint8"))

        else:
            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(generated_images[index])
                    plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()



def generate_basic_model(IMG_DIM=(128, 128, 3)):
    # create and compile the model
    image_size = IMG_DIM[0]
    assert IMG_DIM[0] == IMG_DIM[1], "only square images are supported"
    model = DiffusionModel(image_size, widths, block_depth)
    # below tensorflow 2.9:
    # pip install tensorflow_addons
    # import tensorflow_addons as tfa
    # optimizer=tfa.optimizers.AdamW
    model.compile(
        optimizer=keras.optimizers.experimental.AdamW(
            learning_rate=HyperParameters.learning_rate, weight_decay=HyperParameters.weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
    )
    # pixelwise mean absolute error is used as loss
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
                        const=True, help="print model summary")
    parser.add_argument("--dataset", type=str, default=None,
                   help="train network")
    parser.add_argument("--shuffle-dataset", action="store_const", const=True, default=False,
                   help="shuffle dataset before training")
    parser.add_argument("--dataset-size", type=int, default=100,
                   help="size of the dataset subset to use during training")
    parser.add_argument("--model-type", type=str, choices=["basic", "medium"],
                   default="basic",
                   help="define the type of CNN to use")
    parser.add_argument("--plot-images", type=int, default=0,
                   help="number of images to plot and save at the end of training")

    args = parser.parse_args()

    RESIZED_DIM = 64, 64
    COLOR_DIM = RESIZED_DIM + (3,)

    if args.dataset is None:
        sample_array = []
    else:
        sample_array = load_samples(args.dataset, resized_dim=RESIZED_DIM, size=args.dataset_size)
        print("len of sample array: {} elt(s)".format(len(sample_array)))


    if args.model_type == "basic":
        model = generate_basic_model(COLOR_DIM)
    else:
        raise NotImplementedError

    if args.reset_model:
        pass
    else:
        checkpoint_path = "checkpoints/diffusion_model"
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.restore(checkpoint_path)
        # model = keras.models.load_model(args.model_path)


    # plotting model
    # TODO/FIXME: error on graphviz/pydot import
    # keras.utils.plot_model(model, 'diffusion_model.png', show_shapes=True)

    if args.shuffle_dataset:
        random.shuffle(sample_array)

    # sample_array must have a length which is a multiple of the batch_size
    rem_size = len(sample_array) % HyperParameters.batch_size

    sample_array = [preprocess_image(img, RESIZED_DIM) for img in sample_array[:-rem_size]]

    # preparing training samples (outside of train block to be able to
    # inject them into the summary)
    if not args.train is None and not args.dataset is None:
        val_size = HyperParameters.batch_size
        input_dataset = np.stack([img for img in sample_array])
        train_dataset = input_dataset[:-val_size]
        val_dataset   = input_dataset[-val_size:]
        print(f"len(val_dataset)={len(val_dataset)}")
        print(f"len(train_dataset)={len(train_dataset)}")

        # save the best model based on the validation KID metric
        checkpoint_path = "checkpoints/diffusion_model"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor="val_kid",
            mode="min",
            save_best_only=True,
        )

        # calculate mean and variance of training dataset for normalization
        model.normalizer.adapt(train_dataset)

        # run training and plot generated images periodically
        model.fit(
            train_dataset,
            batch_size=HyperParameters.batch_size,
            epochs=args.train,
            validation_data=(val_dataset,),
            callbacks=[
                # keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
                checkpoint_callback,
            ],
        )
        model.plot_images()
        for i in range(args.plot_images):
            model.plot_images(saveFile=f"hallucinated-landscape-{i}")


    if args.print_model_summary:
        print(model.summary())
