# Title: checkers_v6.py
# Author: Chris Larson
# CS-6700 Final Project

"""This is a checkers engine that generates moves using a convolutional neural network that has been trained
on ~23k masters level checkers games that were recorded from checkers competitions that took place in the 1800 & 1900's.
These games are contained in the text file 'OCA_2.0.pdn', and were parsed and encoded using parser_v7.py. The CNN is
trained using train_v6.py. The model parameters are stored in a checkpoint folder located in the 'parameters' directory."""

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
