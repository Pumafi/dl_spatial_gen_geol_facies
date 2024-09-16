import tensorflow as tf
import numpy as np
from models.ddm_mkj.diffusion_unet import *
from noising_markov_jump_process.noising_process import *


class DiffusionModel(keras.Model):
    def __init__(self, image_size,
                 categories_nb,
                 block_depth=2,
                 batch_size=30,
                 first_conv_channels=64,
                 large_model=False):
        super().__init__()

        self.nb_categories = categories_nb
        self.min_t = 0.001
        self.max_t = .999
        self.step_size = 0.001
        self.steps = np.arange(self.min_t, self.max_t, self.step_size, dtype=np.float64)

        if large_model:
            widths = [64, 128, 256, 512]
            has_attention = [False, False, True, True]
        else:
            widths = [64, 96, 128, 256]
            has_attention = [False, False, True, True]

        # Build the unet model
        self.network = build_model(
            img_size=image_size,
            img_channels=categories_nb,
            widths=widths,
            has_attention=has_attention,
            num_res_blocks=block_depth,
            norm_groups=8,
            activation_fn=keras.activations.swish,
            first_conv_channels=first_conv_channels
        )

        self.ema_network = keras.models.clone_model(self.network)
        self.ema = 0.999

        self.image_size = image_size
        self.batch_size = batch_size

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.post_loss_tracker = keras.metrics.Mean(name="post_loss")

    def conditional_reverse_probability(self, x0, xt, t, tm1, sample=False):
        t = tf.squeeze(t)
        tm1 = tf.squeeze(tm1)

        return reverse_process(x0, xt, t, tm1, self.nb_categories, sample=sample)

    def apply_forward(self, xt, t):
        t = tf.squeeze(t)
        return forward_process(xt, t, self.nb_categories)

    @property
    def metrics(self):
        return [self.image_loss_tracker, self.post_loss_tracker]

    def denoise(self, noisy_images, time, training):
        # the exponential moving average weights are used at evaluation

        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_images = network([noisy_images, time], training=training)

        return pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise

        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            input_diffusion_times = (tf.ones((num_images, 1, 1, 1))) - step * step_size

            pred_images = self.denoise(
                noisy_images, input_diffusion_times, training=False
            )
            next_diffusion_times = input_diffusion_times - step_size

            next_noisy_images, _, _, _ = tf.map_fn(
                lambda x: self.conditional_reverse_probability(x[0], x[1], x[2], x[3], sample=True),
                (pred_images, noisy_images, input_diffusion_times, next_diffusion_times))

        return next_noisy_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        init_x = tf.random.uniform((num_images, 64, 128), 0, 4, dtype=tf.dtypes.int32)
        init_x = keras.utils.to_categorical(init_x)
        generated_images = self.reverse_diffusion(init_x, diffusion_steps)
        return generated_images

    def train_step(self, images):

        # sample uniform random diffusion times t
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.10, maxval=0.9, dtype=tf.float64
        )
        # sample uniform random diffusion times (t - 1)
        diffusion_times_tm1 = tf.clip_by_value(diffusion_times - tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=1e-2, maxval=.5, dtype=tf.float64
        ), 0., 1. - 1e-2)

        noisy_images, _ = tf.map_fn(lambda x: self.apply_forward(x[0], x[1]), (images, diffusion_times))

        pxt, _, _, _ = tf.map_fn(lambda x: self.conditional_reverse_probability(x[0], x[1], x[2], x[3]),
                                 (images, noisy_images, diffusion_times, diffusion_times_tm1))

        with tf.GradientTape() as tape:

            pred_images = self.denoise(noisy_images, diffusion_times, training=True)

            image_loss = self.loss(images, pred_images)  # training loss
            pred_pxt, _, _, _ = tf.map_fn(lambda x: self.conditional_reverse_probability(x[0], x[1], x[2], x[3]),
                                          (pred_images, noisy_images, diffusion_times, diffusion_times_tm1))
            posterior_loss = tf.keras.losses.KLDivergence()(pxt, pred_pxt)
            loss = posterior_loss + 1e-5 * image_loss

        gradients_model = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients_model, self.network.trainable_weights))

        self.image_loss_tracker.update_state(image_loss)
        self.post_loss_tracker.update_state(posterior_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.10, maxval=0.9, dtype=tf.float64
        )
        diffusion_times_tm1 = tf.clip_by_value(diffusion_times - tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=1e-2, maxval=.5, dtype=tf.float64
        ), 0., 1. - 1e-2)

        noisy_images, _ = tf.map_fn(lambda x: self.apply_forward(x[0], x[1]), (images, diffusion_times))
        pxt, _, _, _ = tf.map_fn(lambda x: self.conditional_reverse_probability(x[0], x[1], x[2], x[3]),
                                 (images, noisy_images, diffusion_times, diffusion_times_tm1))

        # use the network to separate noisy images to their components
        pred_images = self.denoise(
            noisy_images, diffusion_times, training=False
        )
        pred_pxt, _, _, _ = tf.map_fn(lambda x: self.conditional_reverse_probability(x[0], x[1], x[2], x[3]),
                                      (pred_images, noisy_images, diffusion_times, diffusion_times_tm1))
        posterior_loss = tf.keras.losses.KLDivergence()(pxt, pred_pxt)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.post_loss_tracker.update_state(posterior_loss)

        return {m.name: m.result() for m in self.metrics}
