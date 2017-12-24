# Original author: Chris Larson
# All Rights Reserved (2016)


import functools
import numpy as np
import pandas as pd
import tensorflow as tf

import checkers


AI1_MODEL = 'parameters/saved_model/step-01001'
AI2_MODEL = 'parameters/saved_model/step-01001'


def ai1_move(predictor, board):
    return board.move_ai('white', predictor)


def ai2_move(predictor, board):
    return board.move_ai('black', predictor)


def play():
    ai1_predictor = tf.contrib.predictor.from_saved_model(AI1_MODEL,
            signature_def_key=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    ai2_predictor = tf.contrib.predictor.from_saved_model(AI2_MODEL,
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
    play()
