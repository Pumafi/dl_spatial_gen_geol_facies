from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import math


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, name="AttentionBlock", **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups, name=name+"-Norm")
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0), name=name+"-Query")
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0), name=name+"-Key")
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0), name=name+"-Value")
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0), name=name+"-Proj")

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, name="TimeEmbedding", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish, dropout=True, name="ResBlock"):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0), name=name+"-Conv0"
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0), name=name+"-Dense1")(temb)[
            :, None, None, :
        ]

        x = layers.GroupNormalization(groups=groups, name=name+"-GN1")(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0), name=name+"-Conv1"
        )(x)
        if dropout:
            x = layers.Dropout(0.3, name=name+"-Dropout1")(x)

        x = layers.Add(name=name+"-Add1")([x, temb])
        x = layers.GroupNormalization(groups=groups, name=name+"-GN2")(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0), name=name+"-Conv2"
        )(x)
        x = layers.Add(name=name+"-Add2")([x, residual])
        return x

    return apply


def DownSample(width, name="DownSample"):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0), name=name+"-Conv"
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest", name="UpSample"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation, name=name+"-UpSamp")(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0), name=name+"-Conv"
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0), name="TimeMLPDense1"
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0), name="TimeMLPDense2")(temb)
        return temb

    return apply