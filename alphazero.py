import numpy as np
import tensorflow as tf


BATCH_SIZE = 1

BOARD_HEIGHT = 8
BOARD_WIDTH = 4

HISTORY = 8

# P1 pieces (normal and kings), P2 pieces (normal and kings), current color
INPUT_PLANES = 5

# We concatenate the historical feature planes with the remainder of the
# features.
INPUT_CHANNELS = HISTORY * INPUT_PLANES

# Four move directions and pass.
OUTPUT_PLANES = 5

CONVOLUTION_CHANNELS = 256


def deepnet():
    # Input
    x = tf.placeholder(tf.float32,
            shape=[BATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS])

    # Output
    #y = tf.placeholder(tf.float32,
    #        shape=[BATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, OUTPUT_PLANES])

    # Variables
    w1 = tf.Variable(tf.truncated_normal([2, 2, INPUT_CHANNELS, CONVOLUTION_CHANNELS], stddev=0.1), name='w1')

    def model(xtrain):
        c1 = tf.nn.conv2d(xtrain, w1, strides=[1, 1, 1, 1], padding='SAME')

        return c1

    y = model(x)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        xin = np.random.rand(BATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS)
        print("input: %s" % xin)

        yout = session.run([y], feed_dict={x: xin})
        print("output: %s" % yout)
        print("shape: %s" % (yout[0].shape,))


if __name__ == "__main__":
    deepnet()
