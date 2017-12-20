# Original author: Chris Larson
# All Rights Reserved (2016)

import numpy as np
import pandas as pd

import checkers


AI_PARAMS = 'parameters/convnet_150k_full/model.ckpt-150001'


def ai_move(board):
    return board.move_ai('black', AI_PARAMS)


def human_move(board):
    print('\n')
    print('  32  31  30  29')
    print('28  27  26  25')
    print('  24  23  22  21')
    print('20  19  18  17')
    print('  16  15  14  13')
    print('12  11  10  09')
    print('  08  07  06  05')
    print('04  03  02  01')

    move_illegal = True
    while move_illegal:
        # Prompt player for input
        move = input("Enter move as 'pos_init, pos_final':")

        if move[::-1][:4][::-1] == 'wins':
            winner = move[0:len(move) - 5]
            end_game = True
            break

        elif move == 'draw':
            winner = 'draw'
            end_game = True
            break

        else:
            move = move.split(',')

            for i in range(len(move) - 1):
                pos_init = int(move[i])
                pos_final = int(move[i + 1])
                if abs(pos_final - pos_init) > 5:
                    move_type = 'jump'
                else:
                    move_type = 'standard'

                # Human positions index from 1, but update excepts an index from 0..
                move_illegal = board.update(positions=[pos_init-1, pos_final-1], player_type='white', move_type=move_type)
                if move_illegal:
                    print('That move is invalid, please try again.')
                else:
                    print("White move: %s" % [pos_init, pos_final])

    return True


def play():
    print('====================================================================================================================================================')
    print('CNN Checkers Engine')
    print('Created by Chris Larson')
    print('All Rights Reserved (2016)')
    print('\n')

    print('You are playing the white pieces, computer is playing black.')
    print('\n')
    print('Procedure:')
    print('1. The computer generates its own moves and prints them to the screen. The user can execute these moves in an external program.')
    print("2. The computer will then prompt the user to enter a sequence of board positions separated by a comma, indicating the move they want to make.")
    print("For example, the entry '7, 10' would indicate moving the piece located at position 7 to position 10. The entry '7, 14, 23' would")
    print("indicate a mulitiple-jump move by the piece located at position 7 eliminating the opposing checkers at positions 10 and 18.")
    print("3. To end the game, specify the result as follows: 'black wins' for a black win, 'white wins' for a white win, or 'draw' for a draw.")
    print('\n')

    checkers.play(human_move, ai_move)


if __name__ == '__main__':
    play()
