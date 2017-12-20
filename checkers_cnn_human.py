# Title: checkers_v6.py
# Author: Chris Larson
# CS-6700 Final Project
# All Rights Reserved (2016)

"""This is a checkers engine that generates moves using a convolutional neural network that has been trained
on ~23k masters level checkers games that were recorded from checkers competitions that took place in the 1800 & 1900's.
These games are contained in the text file 'OCA_2.0.pdn', and were parsed and encoded using parser_v7.py. The CNN is
trained using train_v6.py. The model parameters are stored in a checkpoint folder located in the 'parameters' directory."""

import numpy as np
import pandas as pd

import checkers


AI_PARAMS = 'parameters/convnet_150k_full/model.ckpt-150001'


def play():

    # Alpha-numeric encoding of player turn: white = 1, black = -1
    turn = -1

    # Count number of invalid move attempts
    invalid_move_attempts = 0
    jumps_not_predicted = 0
    move_count = 1
    game_aborted = False

    # Initialize board object
    board = checkers.Board()

    print('====================================================================================================================================================')
    print('CNN Checkers Engine')
    print('Created by Chris Larson')
    print('All Rights Reserved (2016)')
    print('\n')
    print('You are playing the white pieces, computer is playing black.')
    print('There is no GUI for this game. Feel free to run an external program in 2-player mode alongside this game.')
    print('\n')
    print('Procedure:')
    print('1. The computer generates its own moves and prints them to the screen. The user can execute these moves in an external program.')
    print("2. The computer will then prompt the user to enter a sequence of board positions separated by a comma, indicating the move they want to make.")
    print("For example, the entry '7, 10' would indicate moving the piece located at position 7 to position 10. The entry '7, 14, 23' would")
    print("indicate a mulitiple-jump move by the piece located at position 7 eliminating the opposing checkers at positions 10 and 18.")
    print("3. To end the game, specify the result as follows: 'black wins' for a black win, 'white wins' for a white win, or 'draw' for a draw.")
    print('\n')

    # Start game
    input("To begin, press Enter:")
    end_game = False
    winner = ''
    while True:

        # White turn
        if turn == 1:

            print('\n' * 2)
            print('=======================================================')
            print("White's turn")
            print('  32  31  30  29')
            print('28  27  26  25')
            print('  24  23  22  21')
            print('20  19  18  17')
            print('  16  15  14  13')
            print('12  11  10  09')
            print('  08  07  06  05')
            print('04  03  02  01')
            board.print_board()
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

        # Black turn
        elif turn == -1:
            print('\n' * 2)
            print('=======================================================')
            print("Black's turn")
            board.print_board()
            player_type = 'black'

            # Call model to generate move
            moves_list, probs = board.generate_move(player_type=player_type, output_type='top-10', params_dir=AI_PARAMS)
            print(np.array(moves_list) + 1)
            print(probs)

            # Check for available jumps, cross check with moves
            available_jumps = board.find_jumps(player_type=player_type)

            first_move = True

            # Handles situation where there is a jump available to black
            if len(available_jumps) > 0:

                move_type = 'jump'
                jump_available = True

                while jump_available:

                    # For one jump available
                    if len(available_jumps) == 1:
                        count = 1
                        move_predicted = False

                        for move in moves_list:
                            if move == available_jumps[0]:
                                # print("There is one jump available. This move was choice %d." % count)
                                move_predicted = True
                                break
                            else:
                                count += 1

                        if not move_predicted:
                            # print('Model did not output the available jumps. Forced move.')
                            jumps_not_predicted += 1

                        initial_position = available_jumps[0][0]
                        if not (first_move or final_position == initial_position):
                            break
                        final_position = available_jumps[0][1]
                        initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                        move_illegal = board.update(available_jumps[0], player_type=player_type, move_type=move_type)

                        if move_illegal:
                            # print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[0]) + 1))
                            game_aborted = True
                        else:
                            print("Black move: %s" % (np.array(available_jumps[0]) + 1))
                            available_jumps = board.find_jumps(player_type=player_type)
                            final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                            if len(available_jumps) == 0 or final_piece != initial_piece:
                                jump_available = False

                    # When diffent multiple jumps are available
                    else:
                        move_predicted = False
                        for move in moves_list:
                            if move in available_jumps:

                                initial_position = move[0]
                                if not (first_move or final_position == initial_position):
                                    break
                                final_position = move[1]
                                initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                                move_illegal = board.update(move, player_type=player_type, move_type=move_type)

                                if move_illegal:
                                    pass
                                    # print('Model and Find jumps function predicted an invalid move: %s' % (np.array(move) + 1))
                                else:
                                    print("Black move: %s" % (np.array(move) + 1))
                                    move_predicted = True
                                    available_jumps = board.find_jumps(player_type=player_type)
                                    final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                                    if len(available_jumps) == 0 or final_piece != initial_piece:
                                        jump_available = False
                                    break

                        if not move_predicted:
                            # print('Model did not output any of the available jumps. Move picked randomly among valid options.')
                            jumps_not_predicted += 1
                            ind = np.random.randint(0, len(available_jumps))

                            initial_position = available_jumps[ind][0]
                            if not (first_move or final_position == initial_position):
                                break
                            final_position = available_jumps[ind][1]
                            initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                            move_illegal = board.update(available_jumps[ind], player_type=player_type, move_type=move_type)

                            if move_illegal:
                                # print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[ind]) + 1))
                                game_aborted = True
                            else:
                                available_jumps = board.find_jumps(player_type=player_type)
                                final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                                if len(available_jumps) == 0 or final_piece != initial_piece:
                                    jump_available = False

                    first_move = False

            # For standard moves
            else:
                move_type = 'standard'
                move_illegal = True
                while move_illegal:

                    count = 1
                    for move in moves_list:

                        move_illegal = board.update(move, player_type=player_type, move_type=move_type)

                        if move_illegal:
                            # print('model predicted invalid move (%s)' % (np.array(move) + 1))
                            print(probs[count - 1])
                            invalid_move_attempts += 1
                            count += 1
                        else:
                            print('Black move: %s' % (np.array(move) + 1))
                            break

                    if move_illegal:
                        game_aborted = True
                        print("The model failed to provide a valid move. Game aborted.")
                        print(np.array(moves_list) + 1)
                        print(probs)
                        break

        if game_aborted:
            print('Game aborted.')
            break

        if end_game:
            print('The game has ended')
            break
        move_count += 1
        turn *= -1

    # Print out game stats
    end_board = board.state.as_matrix()
    num_black_chkr = len(np.argwhere(end_board == black_chkr))
    num_black_king = len(np.argwhere(end_board == black_king))
    num_white_chkr = len(np.argwhere(end_board == white_chkr))
    num_white_king = len(np.argwhere(end_board == white_king))
    if winner == 'draw':
        print('The game ended in a draw.')
    else:
        print('%s wins' % winner)
    print('Total number of moves: %d' % move_count)
    print('Remaining white pieces: (checkers: %d, kings: %d)' % (num_white_chkr, num_white_king))
    print('Remaining black pieces: (checkers: %d, kings: %d)' % (num_black_chkr, num_black_king))


if __name__ == '__main__':
    play()
