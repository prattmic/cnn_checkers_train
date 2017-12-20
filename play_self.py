# Original author: Chris Larson
# All Rights Reserved (2016)


import numpy as np
import pandas as pd

import checkers


AI1_PARAMS = 'parameters/convnet_150k_full/model.ckpt-150001'
AI2_PARAMS = 'parameters/sample_training/model.ckpt-10001'


def ai1_move(board):
    return board.move_ai('white', AI1_PARAMS)


def ai2_move(board):
    return board.move_ai('black', AI2_PARAMS)


def play():
    print('====================================================================================================================================================')
    print('CNN Checkers Engine')
    print('Created by Chris Larson')
    print('All Rights Reserved (2016)')
    print('\n')

    print('AI1 is playing white, AI2 is playing black.')
    print('\n')

    checkers.play(ai1_move, ai2_move)


if __name__ == '__main__':
    play()
