# Title: parser.py
# Author: Chris Larson
# CS-6700 Final Project
# This program trains a convnet using checkers games extracted from 'OCA_2.0.pdn'.

# Comments ========================================================================================
#
# Board positions:
#  00 01 02 03
#  04 05 06 07
#  08 09 10 11
#  12 13 14 15
#  16 17 18 19
#  20 21 22 23
#  24 25 26 27
#  28 29 30 31
#
#
# Initial board:
#  1  1  1  1
#  1  1  1  1
#  1  1  1  1
#  0  0  0  0
#  0  0  0  0
# -1 -1 -1 -1
# -1 -1 -1 -1
# -1 -1 -1 -1
#
#
# Label output:
#   R L U D
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#
# end comments ====================================================================================

import argparse
import numpy as np
import os
import pandas as pd
from six.moves import cPickle as pickle
import tempfile
import tensorflow as tf
import time


def accuracy(preds, labs):
    n = preds.shape[0]
    acc_vec = np.ndarray([n, 1])
    for i in range(n):
        acc_vec[i, 0] = sum(preds[i, :].astype(int) * labs[i, :].astype(int))
        # print(acc_vec[i, 0])
        assert acc_vec[i, 0] in range(0, 6)
    acc_score = list()
    for j in range(1, 6):
        percentage = len(np.argwhere(acc_vec == j)) / float(n)
        acc_score.append(round(percentage, 4))
    return acc_score


def deepnet(num_steps, lambda_loss, dropout_L1, dropout_L2, model_dir, log_dir):
    # Computational graph
    graph = tf.Graph()
    with graph.as_default():
        # Inputs
        tf_xTr = tf.placeholder(tf.float32, shape=[batch_size, board_height, board_width, num_channels])
        tf_yTr = tf.placeholder(tf.float32, shape=[batch_size, label_height * label_width])
        tf_xTe = tf.constant(xTe)
        tf_xTr_full = tf.constant(xTr)

        # Prediction input.
        tf_xP = tf.placeholder(tf.float32, shape=[1, board_height, board_width, num_channels])

        # Variables
        w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='w1')
        b1 = tf.Variable(tf.zeros([depth]), name='b1')
        w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1), name='w2')
        b2 = tf.Variable(tf.zeros([depth]), name='b2')
        w3 = tf.Variable(tf.truncated_normal([board_height * board_width * depth, num_nodes_layer3], stddev=0.1), name='w3')
        b3 = tf.Variable(tf.zeros([num_nodes_layer3]), name='b3')
        w4 = tf.Variable(tf.truncated_normal([num_nodes_layer3, num_nodes_output], stddev=0.1), name='w4')
        b4 = tf.Variable(tf.zeros([num_nodes_output]), name='b4')

        # Train
        def model(xtrain, dropout_switch):

            # First convolutional layer
            c1 = tf.nn.conv2d(xtrain, w1, strides=[1, 1, 1, 1], padding='SAME')
            h1 = tf.nn.relu(c1 + b1)
            h1_out = tf.nn.dropout(h1, 1 - dropout_L1 * dropout_switch)
            # maxpool1 = tf.nn.max_pool(h1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Second convolutional layer
            c2 = tf.nn.conv2d(h1_out, w2, strides=[1, 1, 1, 1], padding='SAME')
            h2 = tf.nn.relu(c2 + b2)
            h2_out = tf.nn.dropout(h2, 1 - dropout_L1 * dropout_switch)
            # maxpool2 = tf.nn.max_pool(h2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Reshape for fully connected layer
            h2_shape = xtrain.get_shape().as_list()
            h2_out_vec = tf.reshape(h2_out, shape=[h2_shape[0], board_height * board_width * depth])

            # First fully connected layer
            y3 = tf.matmul(h2_out_vec, w3) + b3
            h3 = tf.nn.relu(y3)
            h3_out = tf.nn.dropout(h3, 1 - dropout_L2 * dropout_switch)

            # Model output
            return tf.matmul(h3_out, w4) + b4

        logits = model(tf_xTr, 1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_yTr))
        loss += lambda_loss * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))

        tf.summary.histogram("loss", loss)

        # Optimizer (Built into tensor flow, based on gradient descent)
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.01, batch * batch_size, nTr, 0.95, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

        # Predictions for the training, validation, and test data
        preds_Tr = tf.nn.softmax(model(tf_xTr_full, 0))
        preds_Te = tf.nn.softmax(model(tf_xTe, 0))

        # Single prediction output.
        tf_yP = tf.nn.softmax(model(tf_xP, 1))

    # Feed data into the graph, run the model
    with tf.Session(graph=graph) as session:
        def save_model(step):
            path = os.path.join(model_dir, "step-%05d" % step)
            builder = tf.saved_model.builder.SavedModelBuilder(path)

            signature = tf.saved_model.signature_def_utils.build_signature_def(
                    inputs = {'inputs': tf.saved_model.utils.build_tensor_info(tf_xP)},
                    outputs = {'outputs': tf.saved_model.utils.build_tensor_info(tf_yP)},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

            builder.add_meta_graph_and_variables(
                    session, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map = {
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            signature,
                    })

            builder.save()

        # Run model
        tf.global_variables_initializer().run()
        print('Graph initialized ...')

        writer = tf.summary.FileWriter(log_dir, session.graph)
        summaries = tf.summary.merge_all()

        t = time.time()
        for step in range(num_steps):
            offset = (step * batch_size) % (nTr - batch_size)
            batch_data = xTr[offset:(offset + batch_size), :]
            batch_labels = yTr[offset:(offset + batch_size), :]
            feed_dict = {tf_xTr: batch_data, tf_yTr: batch_labels}

            summary, _ = session.run([summaries, optimizer], feed_dict=feed_dict)

            writer.add_summary(summary, global_step=step)

            if step % 5000 == 0:
                l, preds_Train, preds_Test = session.run([loss, preds_Tr, preds_Te], feed_dict=feed_dict)
                # Find max and set to 1, else 0
                for i in range(nTr):
                    ind_Tr = np.argsort(preds_Train[i, :])[::-1][:5]
                    preds_Train[i, :] = 0
                    for j in range(1, 6):
                        preds_Train[i, ind_Tr[j - 1]] = j
                for i in range(nTe):
                    ind_Te = np.argsort(preds_Test[i, :])[::-1][:5]
                    preds_Test[i, :] = 0
                    for j in range(1, 6):
                        preds_Test[i, ind_Te[j - 1]] = j
                acc_Tr = accuracy(preds_Train, yTr)
                acc_Te = accuracy(preds_Test, yTe)

                print('Minibatch loss at step %d: %f' % (step, l))
                print('Training accuracy of top 5 probabilities: %s' % acc_Tr)
                print('Testing accuracy of top 5 probabilities: %s' % acc_Te)
                print('Time consumed: %d minutes' % ((time.time() - t) / 60.))
                save_model(step)

            elif step % 500 == 0:
                print('Step %d complete ...' % step)

        # Save final model.
        save_model(num_steps)

    print('Training complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train checks CNN')
    parser.add_argument('--training-set', default='checkers_library_500.pickle',
                        help='Pickled training data')
    parser.add_argument('--model-dir', help='Model is saved to this directory')
    parser.add_argument('--log-dir', default='/tmp/tensorflow',
                        help='Tensorboard log directory')
    parser.add_argument('--steps', default=150001, type=int,
                        help='Number of training steps.')

    args = parser.parse_args()

    # Random model directory if none passed.
    if not args.model_dir:
        args.model_dir = tempfile.mkdtemp(prefix='model-')

    print('Model directory: %s' % args.model_dir)
    print('Log directory: %s' % args.log_dir)
    print('Training set: %s' % args.training_set)
    print('Training steps: %d' % args.steps)

    # Define batch size for SGD, and network architecture
    batch_size = 128
    num_channels = 1
    patch_size = 2
    depth = 32
    num_nodes_layer3 = 1024
    num_nodes_output = 128
    board_height = 8
    board_width = 4
    label_height = 32
    label_width = 4

    # Extract training data into win_dict, loss_dict, and draw_dict
    p = pd.read_pickle(args.training_set)
    win_dict = p['win_library']
    loss_dict = p['loss_library']
    draw_dict = p['draw_library']
    print('Finished loading data.')

    # Create numpy arrays xTr(nx8x4) and yTr(nx32x4), where n = number of training examples
    data_list = list()
    labels_list = list()
    for dictionary in [win_dict, loss_dict, draw_dict]:
        for key in dictionary:
            data_list.append(dictionary[key][0].as_matrix())
            labels_list.append(dictionary[key][1].as_matrix())
    data = np.reshape(np.array(data_list, dtype=int), (-1, board_height, board_width, num_channels))
    labels = np.array(labels_list, dtype=int)

    # Randomize order since incoming data is structured into win, loss, draw
    n = len(data_list)
    assert n == len(labels_list)
    ind = np.arange(n)
    np.random.shuffle(ind)
    data, labels = data[ind, :, :], labels[ind, :, :]

    # Vectorize the inputs and labels
    data = data.reshape((-1, board_height, board_width)).astype(np.float32)
    labels = labels.reshape((-1, label_height * label_width)).astype(np.float32)

    # Split x, y into training, cross validation, and test sets
    test_split = 0.35
    nTe = int(test_split * n)
    nTr = n - nTe
    xTe, yTe = data[:nTe, :, :], labels[:nTe, :]
    xTr, yTr = data[nTe:, :, :], labels[nTe:, :]
    assert n == nTr + nTe
    del data, labels

    # Reshape data
    xTr = np.reshape(xTr, (-1, board_height, board_width, num_channels))
    xTe = np.reshape(xTe, (-1, board_height, board_width, num_channels))

    deepnet(num_steps=args.steps,
            lambda_loss=0,
            dropout_L1=0,
            dropout_L2=0,
            model_dir=args.model_dir,
            log_dir=args.log_dir)
