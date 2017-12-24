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


# TODO(prattmic): Don't recreate the predictor each time.
def predict_cnn(board, output, params_dir):
    n = 1
    num_channels = 1
    board_height = 8
    board_width = 4
    label_height = 32
    label_width = 4

    predictor = tf.contrib.predictor.from_saved_model(params_dir,
            signature_def_key=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    # Predict
    board = np.reshape(board.as_matrix(), (n, board_height, board_width, num_channels))
    y = predictor({'inputs': board})['outputs']
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
