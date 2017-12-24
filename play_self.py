# Original author: Chris Larson
# All Rights Reserved (2016)


import argparse
import functools
import numpy as np
import pandas as pd
import tensorflow as tf

import checkers


def ai1_move(predictor, board):
    return board.move_ai('white', predictor)


def ai2_move(predictor, board):
    return board.move_ai('black', predictor)


def play(model1, model2):
    ai1_predictor = tf.contrib.predictor.from_saved_model(model1,
            signature_def_key=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    ai2_predictor = tf.contrib.predictor.from_saved_model(model2,
            signature_def_key=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    print('====================================================================================================================================================')
    print('CNN Checkers Engine')
    print('Created by Chris Larson')
    print('All Rights Reserved (2016)')
    print('\n')

    print('AI1 is playing white, AI2 is playing black.')
    print('\n')

    checkers.play(functools.partial(ai1_move, ai1_predictor),
            functools.partial(ai2_move, ai2_predictor))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checkers self-play')
    parser.add_argument('model1', metavar='MODEL1_PATH', help='AI1 model path')
    parser.add_argument('model2', metavar='MODEL2_PATH', help='AI2 model path')

    args = parser.parse_args()

    play(args.model1, args.model2)
