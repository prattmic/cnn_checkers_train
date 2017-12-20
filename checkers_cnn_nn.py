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


def play():

    # Alpha-numeric encoding of player turn: AI1 = 1, AI2 = -1
    turn = -1

    move_count = 1

    # Initialize board object
    board = checkers.Board()

    print('====================================================================================================================================================')
    print('CNN Checkers Engine')
    print('Created by Chris Larson')
    print('\n')
    print('AI1 is playing white, AI2 is playing black.')
    print('There is no GUI for this game. Feel free to run an external program in 2-player mode alongside this game.')
    print('\n')

    # Start game
    input("To begin, press Enter:")

    while True:

        # White turn
        if turn == 1:
            print('\n' * 2)
            print('=======================================================')
            player_type = 'white'
            print('Move %d: %s' % (move_count, player_type))
            board.print_board()
            params_dir = AI1_PARAMS

        # Black turn
        else:
            print('\n' * 2)
            print('=======================================================')
            player_type = 'black'
            print('Move %d: %s' % (move_count, player_type))
            board.print_board()
            params_dir = AI2_PARAMS

        game_aborted = not board.move_ai(player_type, params_dir)

        # Check game status
        num_black_pieces = len(np.argwhere(board.state.as_matrix() > 0))
        num_white_pieces = len(np.argwhere(board.state.as_matrix() < 0))

        if num_black_pieces == 0:
            winner = 'white'
            break
        elif num_white_pieces == 0:
            winner = 'black'
            break
        elif move_count >= 100:
            winner = 'draw'
            break
        elif game_aborted:
            winner = 'n/a'
            break

        move_count += 1
        turn *= -1

    # Print out game stats
    end_board = board.state.as_matrix()
    print('Ending board:')
    print(board.board_state(player_type='white'))
    num_black_chkr = len(np.argwhere(end_board == checkers.BLACK_CHECKER))
    num_black_king = len(np.argwhere(end_board == checkers.BLACK_KING))
    num_white_chkr = len(np.argwhere(end_board == checkers.WHITE_CHECKER))
    num_white_king = len(np.argwhere(end_board == checkers.WHITE_KING))

    if winner == 'draw':
        print('The game ended in a draw.')
    else:
        print('%s wins' % winner)

    print('Total number of moves: %d' % move_count)
    print('Remaining white pieces: (checkers: %d, kings: %d)' % (num_white_chkr, num_white_king))
    print('Remaining black pieces: (checkers: %d, kings: %d)' % (num_black_chkr, num_black_king))
    print('Invalid move attempts: %d' % board.invalid_move_attempts)
    print('Jumps not predicted: %d' % board.jumps_not_predicted)


if __name__ == '__main__':
    play()
