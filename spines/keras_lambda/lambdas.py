import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class AddAxis(tf.keras.layers.Layer):
    """ Add Axis layer. """

    def __init__(self, axis=-1):
        super(AddAxis, self).__init__()
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


class TimesNum(tf.keras.layers.Layer):
    """ times nums layer. """

    def __init__(self, level_digit):
        super(TimesNum, self).__init__()
        self.level_digit = level_digit

    def __call__(self, inputs):
        return inputs * self.level_digit


class Tanh(tf.keras.layers.Layer):
    """ Use tanh layer. """

    def __init__(self):
        super(Tanh, self).__init__()

    def call(self, inputs):
        return tf.math.tanh(inputs)


class Squeeze(tf.keras.layers.Layer):
    """ Use squeeze layer. """

    def __init__(self):
        super(Squeeze, self).__init__()

    def call(self, inputs):
        return tf.squeeze(inputs)

