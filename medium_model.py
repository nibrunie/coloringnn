import math

from tensorflow import keras
from tensorflow.keras import layers

def generate_coloring_medium_model(GRAY_DIM=(128, 128, 1)):
    """ generate a very basic coloring model which takes as input a
        gray image and try to colorize it """
    inputs = keras.Input(shape=(GRAY_DIM), name='gray_image')
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2d_1')(inputs)
    conv2d_2 = layers.Conv2D(64, 3, activation='relu', padding='same',name='conv2d_2')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_1")(conv2d_2)
    x = layers.Conv2D(128, 3, activation='relu', padding='same',name='conv2d_3')(x)
    x = layers.BatchNormalization(axis=1)(x)
    conv2d_4 = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_2")(conv2d_4)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_5')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_6')(x)
    x = layers.BatchNormalization(axis=1)(x)
    conv2d_7 = layers.Conv2D(128, 3, activation='relu', name='conv2d_7')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_3")(conv2d_7)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_8')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_9')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_10')(x)
    x = layers.BatchNormalization(axis=1)(x)

    x = layers.Conv2D(128, 1, activation="relu", padding='same', name="conv2d_1x1")(x)
    x = layers.UpSampling2D(name="up_sampling_1")(x)
    x = layers.Add()([x, conv2d_7])
    x = layers.Conv2D(64, 3, activation="relu", padding='same', name="conv2d_recons_2")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Conv2DTranspose(128, 3, name="conv2d_transpose_1")(x)
    x = layers.UpSampling2D(name="up_sampling_2")(x)
    x = layers.Add()([x, conv2d_4])
    x = layers.Conv2D(64, 3, activation="relu", padding='same', name="conv2d_recons_3")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.UpSampling2D(name="up_sampling_3")(x)
    x = layers.Add()([x, conv2d_2])
    x = layers.Conv2D(3, 3, activation="relu", padding='same', name="conv2d_recons_4")(x)

    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.RMSprop())

    return model

def generate_super_reso_medium_model(INPUT_DIM=(64, 64, 3), SUPER_DIM=(512, 512, 3)):
    """ generate a very basic coloring model which takes as input a
        gray image and try to colorize it """
    upLevels = int(math.log2(SUPER_DIM[0] / INPUT_DIM[0]))
    inputs = keras.Input(shape=(INPUT_DIM), name='input_image')
    x = layers.Conv2D(16, 3, activation='relu', padding='same', name='conv2d_1')(inputs)
    conv2d_1 = x
    conv2d_2 = layers.Conv2D(16, 3, activation='relu', padding='same',name='conv2d_2')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_1")(conv2d_2)
    x = layers.Conv2D(32, 3, activation='relu', padding='same',name='conv2d_3')(x)
    x = layers.BatchNormalization(axis=1)(x)
    conv2d_4 = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_2")(conv2d_4)
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2d_5')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2d_6')(x)
    x = layers.BatchNormalization(axis=1)(x)
    conv2d_7 = layers.Conv2D(32, 3, activation='relu', name='conv2d_7')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_3")(conv2d_7)
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2d_8')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2d_9')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2d_10')(x)
    x = layers.BatchNormalization(axis=1)(x)

    x = layers.Conv2D(32, 1, activation="relu", padding='same', name="conv2d_1x1")(x)
    x = layers.UpSampling2D(name="up_sampling_1")(x)
    x = layers.Add()([x, conv2d_7])
    x = layers.Conv2D(16, 3, activation="relu", padding='same', name="conv2d_recons_2")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Conv2DTranspose(32, 3, name="conv2d_transpose_1")(x)
    x = layers.UpSampling2D(name="up_sampling_2")(x)
    x = layers.Add()([x, conv2d_4])
    x = layers.Conv2D(16, 3, activation="relu", padding='same', name="conv2d_recons_3")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.UpSampling2D(name="up_sampling_3")(x)
    x = layers.Add()([x, conv2d_2])
    skip = conv2d_1
    for upLevel in range(upLevels):
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.UpSampling2D(name=f"up_sampling_lvl-{upLevel}")(x)
        skip = layers.UpSampling2D(name=f"up-skip-lvl-{upLevel}")(skip)
        x = layers.Add()([x, skip])
        x = layers.Conv2D(16, 3, activation="relu", padding='same', name=f"conv2d_recons_up-lvl-f{upLevel}")(x)

    x = layers.Conv2D(3, 3, activation="relu", padding='same', name="conv2d_recons_4")(x)

    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)

    #model.compile(loss=keras.losses.MeanSquaredError(),
    #              optimizer=keras.optimizers.RMSprop())

    return model

model = generate_super_reso_medium_model()
print(model.summary())
keras.utils.plot_model(model, 'super_reso_medium_model.png', show_shapes=True)
