import tensorflow as tf
from spines.keras_lambda.lambdas import AddAxis


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv1D(filters=num_channels,
                                           kernel_size=3, padding='causal')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv1D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool1D(pool_size=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)


def block_1(shape):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=shape),
        AddAxis(axis=-1),
        tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='causal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool1D(pool_size=2)])


def block_2(shape):
    net = block_1(shape)
    # `num_channels`为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net


def net(shape, output_nums):
    net = block_2(shape)
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool1D())
    net.add(tf.keras.layers.Flatten())

    net.add(tf.keras.layers.Dense(2048))
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.Dropout(0.1))
    net.add(tf.keras.layers.Dense(1024))
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.Dropout(0.1))

    net.add(tf.keras.layers.Dense(output_nums))

    net.compile(
        metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss=tf.keras.losses.MSE  # MAE
                )
    return net

