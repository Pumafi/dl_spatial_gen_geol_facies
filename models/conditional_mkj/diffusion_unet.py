import tensorflow as tf
from tensorflow import keras

from models.custom_diffusion_layers import *


def build_model(img_size,
                img_channels,
                widths,
                has_attention,
                num_res_blocks=2,
                norm_groups=8,
                interpolation="nearest",
                activation_fn=keras.activations.swish,
                first_conv_channels=64):
    image_width = img_size[0]
    image_height = img_size[1]
    image_input = layers.Input(
        shape=(image_width, image_height, img_channels), name="image_input"
    )
    time_input = keras.Input(shape=(1, 1, 1), dtype=tf.int64, name="time_input")

    # Conditioning Inputs
    mask = layers.Input(shape=(image_width, image_height, 1), name="mask_input")
    pixels = layers.Input(shape=(image_width, image_height, img_channels), name="pixels_input")

    init_conv = layers.Conv2D(first_conv_channels,
                              kernel_size=(1, 1),
                              padding="same",
                              kernel_initializer=kernel_init(1.0),
                              name="InitialConv2D")
    x = init_conv(image_input)
    pixels_x = init_conv(pixels)
    x = tf.keras.layers.Concatenate(axis=-1)([x, pixels_x, mask])
    x = layers.Conv2D(first_conv_channels,
                      kernel_size=(3, 3),
                      padding="same",
                      name="ConditioningConv2D")(x)

    temb = tf.reshape(time_input, (-1,))
    temb = TimeEmbedding(dim=first_conv_channels * 4)(temb)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=keras.layers.LeakyReLU())(temb)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for j in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn, name="ResBlock-Down-{0}-{1}".format(i, j)
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups, name="AttentionBlock-Down-{0}-{1}".format(i, j))(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i], name="DownSample-{0}".format(i))(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn, dropout=False, name="MidResBlock1")(
        [x, temb]
    )
    x = AttentionBlock(widths[-1], groups=norm_groups, name="MidAttBlock")(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn, dropout=False, name="MidResBlock2")(
        [x, temb]
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for j in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1, name="Up-Concat-{0}-{1}".format(i, j))([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn, name="ResBlock-Up-{0}-{1}".format(i, j)
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups, name="AttentionBlock-Up-{0}-{1}".format(i, j))(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation, name="UpSample-{0}".format(i))(x)

    # End block
    # x = layers.GroupNormalization(groups=norm_groups, name="FinalGN")(x)
    x = layers.Conv2D(img_channels, (1, 1), padding="same", kernel_initializer=kernel_init(0.0), activation="softmax",
                      name="FinalConv2D")(x)
    x = tf.cast(x, tf.float64)
    return keras.Model([image_input, time_input, mask, pixels], x, name="unet")