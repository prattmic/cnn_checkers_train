import numpy as np
import tensorflow as tf


def predict_nn(board, output, params_dir):

    # Define batch size for SGD, and network architecture
    n = 1
    num_nodes_layer1 = 512
    num_nodes_layer2 = 512
    board_height = 8
    board_width = 4
    label_height = 32
    label_width = 4

    # Model input
    tf_x = tf.placeholder(tf.float32, shape=[n, board_height * board_width])

    # Start interactive tf session
    session = tf.InteractiveSession()

    # Variables
    w1 = tf.Variable(tf.truncated_normal([board_height * board_width, num_nodes_layer1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([num_nodes_layer1]))
    w2 = tf.Variable(tf.truncated_normal([num_nodes_layer1, num_nodes_layer2], stddev=0.1))
    b2 = tf.Variable(tf.zeros([num_nodes_layer2]))
    w3 = tf.Variable(tf.truncated_normal([num_nodes_layer2, label_height * label_width], stddev=0.1))
    b3 = tf.Variable(tf.zeros([label_height * label_width]))

    # Compute
    y1 = tf.add(tf.matmul(tf_x, w1), b1)
    h1 = tf.nn.relu(y1)
    y2 = tf.add(tf.matmul(h1, w2), b2)
    h2 = tf.nn.relu(y2)
    y3 = tf.add(tf.matmul(h2, w3), b3)
    y_pred = tf.nn.softmax(y3)

    # Restore saved model params
    var_dict = {'w1': w1,
                'b1': b1,
                'w2': w2,
                'b2': b2,
                'w3': w3,
                'b3': b3,
                }
    saver = tf.train.Saver(var_dict)
    init = tf.initialize_all_variables()
    session.run(init)
    saver.restore(session, params_dir)

    # Predict
    board = np.reshape(board.as_matrix(), (-1, board_height * board_width))
    y = y_pred.eval(feed_dict={tf_x: board})
    y = np.reshape(y, (label_height * label_width,))
    norm = np.sum(y)

    # Return max or top-5 prob(s)
    if output == 'top-1':
        ind = np.argmax(y[:])
        y[:] = 0
        y[ind] = 1
    elif output == 'top-5':
        ind = np.argsort(y[:])[::-1][:5]
        probs = y[ind] / norm
        y[:] = 0
        for j in range(1, 6):
            y[ind[j - 1]] = j
    elif output == 'top-10':
        ind = np.argsort(y[:])[::-1][:10]
        probs = y[ind] / norm
        y[:] = 0
        for j in range(1, 11):
            y[ind[j - 1]] = j
    elif output == 'top-50':
        ind = np.argsort(y[:])[::-1][:50]
        probs = y[ind] / norm
        y[:] = 0
        for j in range(1, 51):
            y[ind[j - 1]] = j

    return np.reshape(y, (label_height, label_width)).astype(int), probs


def predict_cnn(board, output, params_dir):

    n = 1
    num_channels = 1
    patch_size = 2
    depth = 32
    num_nodes_layer3 = 1024
    num_nodes_output = 128
    board_height = 8
    board_width = 4
    label_height = 32
    label_width = 4

    # Model input
    tf_x = tf.placeholder(tf.float32, shape=[n, board_height, board_width, num_channels])

    # Start interactive tf session
    session = tf.InteractiveSession()

    # Variables
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.zeros([depth]), name='b1')
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1), name='w2')
    b2 = tf.Variable(tf.zeros([depth]), name='b2')
    w3 = tf.Variable(tf.truncated_normal([board_height * board_width * depth, num_nodes_layer3], stddev=0.1), name='w3')
    b3 = tf.Variable(tf.zeros([num_nodes_layer3]), name='b3')
    w4 = tf.Variable(tf.truncated_normal([num_nodes_layer3, num_nodes_output], stddev=0.1), name='w4')
    b4 = tf.Variable(tf.zeros([num_nodes_output]), name='b4')

    # Compute
    c1 = tf.nn.conv2d(tf_x, w1, strides=[1, 1, 1, 1], padding='SAME')
    h1 = tf.nn.relu(c1 + b1)
    c2 = tf.nn.conv2d(h1, w2, strides=[1, 1, 1, 1], padding='SAME')
    h2 = tf.nn.relu(c2 + b2)
    h2_shape = tf_x.get_shape().as_list()
    h2_out_vec = tf.reshape(h2, shape=[h2_shape[0], board_height * board_width * depth])
    y3 = tf.matmul(h2_out_vec, w3) + b3
    h3 = tf.nn.relu(y3)
    y4 = tf.matmul(h3, w4) + b4
    y_pred = tf.nn.softmax(y4)

    # Restore saved model params
    var_dict = {'w1': w1,
                'b1': b1,
                'w2': w2,
                'b2': b2,
                'w3': w3,
                'b3': b3,
                'w4': w4,
                'b4': b4,
                }
    saver = tf.train.Saver(var_dict)
    init = tf.initialize_all_variables()
    session.run(init)
    saver.restore(session, params_dir)

    # Predict
    board = np.reshape(board.as_matrix(), (n, board_height, board_width, num_channels))
    y = y_pred.eval(feed_dict={tf_x: board})
    norm = np.sum(y)

    # Return max or top-5 prob(s)
    if output == 'one-vs-all':
        for i in range(n):
            ind = np.argmax(y[i, :])
            y[i, :] = 0
            y[i, ind] = 1
    elif output == 'top-5':
        for i in range(n):
            ind = np.argsort(y[i, :])[::-1][:5]
            probs = y[i, ind] / norm
            y[i, :] = 0
            for j in range(1, 6):
                y[i, ind[j - 1]] = j
    elif output == 'top-10':
        for i in range(n):
            ind = np.argsort(y[i, :])[::-1][:10]
            probs = y[i, ind] / norm
            y[i, :] = 0
            for j in range(1, 11):
                y[i, ind[j - 1]] = j
    elif output == 'top-50':
        for i in range(n):
            ind = np.argsort(y[i, :])[::-1][:50]
            probs = y[i, ind] / norm
            y[i, :] = 0
            for j in range(1, 51):
                y[i, ind[j - 1]] = j

    return np.reshape(y, (label_height, label_width)).astype(int), probs