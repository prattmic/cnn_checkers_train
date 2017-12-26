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

RESIDUAL_BLOCKS = 6

LOG_DIR = "/tmp/tensorflow"


def deepnet():
    # Input
    x = tf.placeholder(tf.float32,
            shape=[BATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS])

    # Output
    #y = tf.placeholder(tf.float32,
    #        shape=[BATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, OUTPUT_PLANES])

    def model(x1):
        def residual_block(x1):
            c1 = tf.layers.conv2d(x1, CONVOLUTION_CHANNELS, kernel_size=3, padding='SAME')
            b1 = tf.layers.batch_normalization(c1, axis=3)
            h1 = tf.nn.relu(b1)

            c2 = tf.layers.conv2d(h1, CONVOLUTION_CHANNELS, kernel_size=3, padding='SAME')
            b2 = tf.layers.batch_normalization(c2, axis=3)
            h2 = tf.nn.relu(b2 + x1)

            return h2

        # Input convolution.
        with tf.variable_scope("input"):
            c1 = tf.layers.conv2d(x1, CONVOLUTION_CHANNELS, kernel_size=3, padding='SAME')
            b1 = tf.layers.batch_normalization(c1, axis=3)
            h1 = tf.nn.relu(b1)

        # Residual blocks.
        v = h1
        for i in range(RESIDUAL_BLOCKS):
            with tf.variable_scope("residual_block%d" % i):
                v = residual_block(v)

        # Policy head.
        with tf.variable_scope("policy"):
            pc = tf.layers.conv2d(v, 2, kernel_size=1, padding='SAME')
            pb = tf.layers.batch_normalization(pc, axis=3)
            ph = tf.nn.relu(pb)

            policy = tf.layers.dense(ph, BATCH_SIZE * BOARD_HEIGHT * BOARD_WIDTH * OUTPUT_PLANES)

        # Value head.
        with tf.variable_scope("value"):
            vc = tf.layers.conv2d(v, 1, kernel_size=1, padding='SAME')
            vb = tf.layers.batch_normalization(vc, axis=3)
            vh = tf.nn.relu(vb)

            d1 = tf.layers.dense(vh, 256)
            hidden = tf.nn.relu(d1)

            d2 = tf.layers.dense(hidden, 1)
            value = tf.nn.tanh(d2)

        return policy, value

    policy, value = model(x)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter(LOG_DIR, session.graph)

        xin = np.ones([BATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS])
        print("input: %s" % xin)

        out = session.run([policy, value], feed_dict={x: xin})
        print("output: %s" % out)
        print("shape: %s" % (out[0].shape,))
        print("shape: %s" % (out[1].shape,))


if __name__ == "__main__":
    deepnet()
