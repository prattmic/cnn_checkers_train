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

        # Call model to generate move
        moves_list, probs = board.generate_move(player_type=player_type, output_type='top-10', params_dir=params_dir)
        print(np.array(moves_list) + 1)
        print(probs)

        # Check for available jumps, cross check with moves
        available_jumps = board.find_jumps(player_type=player_type)

        first_move = True

        # Handles situation in which jump is available
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
                            print("There is one jump available. This move was choice %d." % count)
                            move_predicted = True
                            break
                        else:
                            count += 1

                    if not move_predicted:
                        print('Model did not output the available jumps. Forced move.')
                        jumps_not_predicted += 1

                    initial_position = available_jumps[0][0]
                    if not (first_move or final_position == initial_position):
                        break
                    final_position = available_jumps[0][1]
                    initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                    move_illegal = board.update(available_jumps[0], player_type=player_type, move_type=move_type)

                    if move_illegal:
                        print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[0]) + 1))
                        game_aborted = True
                    else:
                        print("%s move: %s" % (player_type, (np.array(available_jumps[0]) + 1)))
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
                                print('Model and Find jumps function predicted an invalid move: %s' % (np.array(move) + 1))
                            else:
                                print("%s move: %s" % (player_type, (np.array(move) + 1)))
                                move_predicted = True
                                available_jumps = board.find_jumps(player_type=player_type)
                                final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                                if len(available_jumps) == 0 or final_piece != initial_piece:
                                    jump_available = False
                                break

                    if not move_predicted:
                        print('Model did not output any of the available jumps. Move picked randomly among valid options.')
                        jumps_not_predicted += 1
                        ind = np.random.randint(0, len(available_jumps))

                        initial_position = available_jumps[ind][0]
                        if not (first_move or final_position == initial_position):
                            break
                        final_position = available_jumps[ind][1]
                        initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                        move_illegal = board.update(available_jumps[ind], player_type=player_type, move_type=move_type)

                        if move_illegal:
                            print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[ind]) + 1))
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
                        print('model predicted invalid move (%s)' % (np.array(move) + 1))
                        print(probs[count - 1])
                        invalid_move_attempts += 1
                        count += 1
                    else:
                        print('%s move: %s' % (player_type, (np.array(move) + 1)))
                        break

                if move_illegal:
                    game_aborted = True
                    print("The model failed to provide a valid move. Game aborted.")
                    print(np.array(moves_list) + 1)
                    print(probs)
                    break

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
    print('Invalid move attempts: %d' % invalid_move_attempts)
    print('Jumps not predicted: %d' % jumps_not_predicted)


if __name__ == '__main__':
    play()
