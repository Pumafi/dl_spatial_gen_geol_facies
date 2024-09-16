import numpy as np
import tensorflow as tf

g = tf.random.get_global_generator()


def cosine_schedule_function(t, max_t=1., epsilon=1e-3):
    t = tf.cast(t, tf.float64)
    f_t = tf.math.cos(((t / max_t) + epsilon) / (1 + epsilon) * (np.pi / 2)) ** 2
    f_0 = tf.math.cos(((t * 0 / max_t) + epsilon) / (1 + epsilon) * (np.pi / 2)) ** 2
    return f_t / f_0


def make_qmatrix_integral(nb_categories, t, tm1=None):
    Q_integral = tf.zeros((nb_categories, nb_categories), dtype=tf.float64)

    if tm1 is None:
        tm1 = tf.zeros_like(t, dtype=tf.float64)

    for i in range(nb_categories):
        for j in range(nb_categories):
            if i != j:
                Q_integral = tf.tensor_scatter_nd_update(Q_integral, indices=tf.constant([[i, j]]),
                                                         updates=[-1 / (nb_categories - 1) * tf.squeeze(
                                                             tf.math.log(cosine_schedule_function(t, max_t=1.)))
                                                                  + 1 / (nb_categories - 1) * tf.squeeze(
                                                             tf.math.log(cosine_schedule_function(tm1, max_t=1.)))])

    for i in range(nb_categories):
        Q_integral = tf.tensor_scatter_nd_update(Q_integral, indices=tf.constant([[i, i]]),
                                                 updates=[-tf.reduce_sum(Q_integral[i])])

    return Q_integral


def probability_function(nb_categories, t, step_size):
    """
    Compute the categorical forward probability from t to t + step_size

    """
    t = tf.cast(t, dtype=tf.float64)
    if step_size == 0.:
        tm1 = None
    else:
        tm1 = tf.cast(t - step_size, dtype=tf.float64)
    Q = make_qmatrix_integral(nb_categories, t, tm1)

    return Q


@tf.function
def forward_process(x0, step, nb_categories):
    if len(x0.shape) != 3:
        raise Exception("Entry must have 3 dims but {0} were found".format(len(x0.shape)))

    t = tf.squeeze(step)
    Q = tf.linalg.expm(make_qmatrix_integral(nb_categories, t))

    pxt = tf.tensordot(x0, Q, axes=[-1, 0])
    height, width, _ = pxt.shape
    flat_pxt = tf.reshape(pxt, (-1, nb_categories))
    cum_sum = tf.math.cumsum(flat_pxt, axis=-1)

    unif = tf.cast(g.uniform(shape=(len(cum_sum), 1)), dtype=tf.float64)
    random_values = tf.math.argmax((unif < cum_sum), axis=1)
    xt = tf.reshape(random_values, (height, width))
    xt = tf.cast(tf.one_hot(xt, nb_categories), dtype=tf.float64)

    return xt, step


@tf.function
def reverse_process(x0, xt, t, tm1, nb_ctg, sample=False):
    t = tf.cast(tf.squeeze(t), dtype=tf.float64)
    tm1 = tf.cast(tf.squeeze(tm1), dtype=tf.float64)

    Q_tm1_t = tf.linalg.expm(make_qmatrix_integral(nb_ctg, t, tm1))

    Q_0_tm1 = tf.linalg.expm(make_qmatrix_integral(nb_ctg, tm1))

    Q_0_t = tf.linalg.expm(make_qmatrix_integral(nb_ctg, t))

    eqt = tf.tensordot(xt, Q_tm1_t, axes=[-1, 0])
    eq0 = tf.tensordot(x0, Q_0_tm1, axes=[-1, 0])

    nominator = tf.multiply(eqt, eq0)
    denom = tf.expand_dims(tf.reduce_sum(tf.multiply(tf.tensordot(x0, Q_0_t, axes=[-1, 0]), xt), axis=-1), axis=-1)
    pxt = tf.clip_by_value(nominator / denom, 0., 1.)
    if sample:
        height, width, _ = pxt.shape
        flat_pxt = tf.reshape(pxt, (-1, nb_ctg))
        cum_sum = tf.math.cumsum(flat_pxt, axis=-1)
        unif = tf.cast(g.uniform(shape=(len(cum_sum), 1)), dtype=tf.float64)
        random_values = tf.math.argmax((unif < cum_sum), axis=1)
        pxt = tf.reshape(random_values, (height, width))
        pxt = tf.one_hot(pxt, nb_ctg)
        pxt = tf.cast(pxt, dtype=tf.float64)

    return pxt, xt, t, tm1

