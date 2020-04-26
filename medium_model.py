from tensorflow import keras
from tensorflow.keras import layers

def generate_medium_model(GRAY_DIM=(128, 128, 1)):
    """ generate a very basic coloring model which takes as input a
        gray image and try to colorize it """
    inputs = keras.Input(shape=(GRAY_DIM), name='gray_image')
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2d_1')(inputs)
    conv2d_2 = layers.Conv2D(64, 3, activation='relu', padding='same',name='conv2d_2')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_1")(conv2d_2)
    x = layers.Conv2D(128, 3, activation='relu', padding='same',name='conv2d_3')(x)
    conv2d_4 = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_2")(conv2d_4)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_5')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_6')(x)
    conv2d_7 = layers.Conv2D(128, 3, activation='relu', name='conv2d_7')(x)
    x = layers.MaxPool2D((2,2),name="maxpool_3")(conv2d_7)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_8')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_9')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_10')(x)

    x = layers.Conv2D(128, 1, activation="relu", padding='same', name="conv2d_1x1")(x)
    x = layers.UpSampling2D(name="up_sampling_1")(x)
    x = layers.Add()([x, conv2d_7])
    x = layers.Conv2D(64, 3, activation="relu", padding='same', name="conv2d_recons_2")(x)
    x = layers.Conv2DTranspose(128, 3, name="conv2d_transpose_1")(x)
    x = layers.UpSampling2D(name="up_sampling_2")(x)
    x = layers.Add()([x, conv2d_4])
    x = layers.Conv2D(64, 3, activation="relu", padding='same', name="conv2d_recons_3")(x)
    x = layers.UpSampling2D(name="up_sampling_3")(x)
    x = layers.Add()([x, conv2d_2])
    x = layers.Conv2D(3, 3, activation="relu", padding='same', name="conv2d_recons_4")(x)

    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.RMSprop())

    return model


#model = generate_medium_model()
#print(model.summary())
#keras.utils.plot_model(model, 'medium_model.png', show_shapes=True)
