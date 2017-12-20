# Title: checkers_v6.py
# Author: Chris Larson
# CS-6700 Final Project

"""This is a checkers engine that generates moves using a convolutional neural network that has been trained
on ~23k masters level checkers games that were recorded from checkers competitions that took place in the 1800 & 1900's.
These games are contained in the text file 'OCA_2.0.pdn', and were parsed and encoded using parser_v7.py. The CNN is
trained using train_v6.py. The model parameters are stored in a checkpoint folder located in the 'parameters' directory."""

import numpy as np
import pandas as pd
import predict_move


AI1_PARAMS = 'parameters/convnet_150k_full/model.ckpt-150001'
AI2_PARAMS = 'parameters/sample_training/model.ckpt-10001'


class Board(object):

    global jumps, empty, odd_list

    def __init__(self):
        self.state = pd.read_csv(filepath_or_buffer='board_init.csv', header=-1, index_col=None)
        self.invalid_attempts = 0

    def board_state(self, player_type):
        if player_type == 'white':
            return -self.state.iloc[::-1, ::-1]
        elif player_type == 'black':
            return self.state

    def print_board(self):
        for j in range(8):
            for i in range(4):
                if j % 2 == 0:
                    print(' ', end=' ')
                if self.state[3 - i][7 - j] == 1:
                    print('x', end=' ')
                elif self.state[3 - i][7 - j] == 3:
                    print('X', end=' ')
                elif self.state[3 - i][7 - j] == 0:
                    print('-', end=' ')
                elif self.state[3 - i][7 - j] == -1:
                    print('o', end=' ')
                else:
                    print('O', end=' ')
                if j % 2 != 0:
                    print(' ', end=' ')
            print('')

    def find_jumps(self, player_type):

        valid_jumps = list()

        if player_type == 'black':
            king_value = black_king
            chkr_value = black_chkr
            chkr_directions = [1, 2]
        else:
            king_value = white_king
            chkr_value = white_chkr
            chkr_directions = [0, 3]

        board_state = self.state.copy()
        board_state = np.reshape(board_state.as_matrix(), (32,))

        for position in range(32):
            piece = board_state[position]
            neighbors_list = neighbors[position]
            next_neighbors_list = next_neighbors[position]

            if piece == chkr_value:
                for direction in chkr_directions:
                    neighbor = neighbors_list[direction]
                    next_neighbor = next_neighbors_list[direction]
                    if neighbor == iv or next_neighbor == iv:
                        pass
                    elif board_state[next_neighbor] == empty and (board_state[neighbor] == -chkr_value or board_state[neighbor] == -king_value):
                        valid_jumps.append([position, next_neighbor])

            elif piece == king_value:
                for direction in range(4):
                    neighbor = neighbors_list[direction]
                    next_neighbor = next_neighbors_list[direction]
                    if neighbor == iv or next_neighbor == iv:
                        pass
                    elif board_state[next_neighbor] == empty and (board_state[neighbor] == -chkr_value or board_state[neighbor] == -king_value):
                        valid_jumps.append([position, next_neighbor])

        return valid_jumps

    def get_positions(self, move, player_type):

        # Extract starting position, and direction to move
        ind = np.argwhere(move == 1)[0]
        position = ind[0]
        direction = ind[1]

        jumps_available = self.find_jumps(player_type=player_type)

        neighbor = neighbors[position][direction]
        next_neighbor = next_neighbors[position][direction]

        if [position, next_neighbor] in jumps_available:
            return position, next_neighbor, 'jump'
        else:
            return position, neighbor, 'standard'

    def generate_move(self, player_type, output_type, params_dir):

        board_state = self.board_state(player_type=player_type)
        moves_list = list()

        # Assumes both AI are the same CNN model.
        moves, probs = predict_move.predict_cnn(board_state, output=output_type, params_dir=params_dir)

        for i in range(1, 11):
            ind = np.argwhere(moves == i)[0]
            move = np.zeros([32, 4])
            move[ind[0], ind[1]] = 1
            if player_type == 'white':
                move = move[::-1, :]
                move = np.concatenate((move[:, 2:], move[:, :2]), axis=1)
            pos_init, pos_final, move_type = self.get_positions(move, player_type=player_type)
            moves_list.append([pos_init, pos_final])

            # # If white, flip board back
            # if player_type == 'white':
            #     for move in moves_list:
            #         move[0] = 31 - move[0]
            #         move[1] = 31 - move[1]

        return moves_list, probs

    def update(self, positions, player_type, move_type):

        # Extract the initial and final positions into ints
        [pos_init, pos_final] = int(positions[0]), (positions[1])

        if player_type == 'black':
            king_pos = black_king_pos
            king_value = black_king
            chkr_value = black_chkr

        else:
            king_pos = white_king_pos
            king_value = white_king
            chkr_value = white_chkr

        # print(pos_init, pos_final)
        board_vec = self.state.copy()
        board_vec = np.reshape(board_vec.as_matrix(), (32,))

        if (board_vec[pos_init] == chkr_value or board_vec[pos_init] == king_value) and board_vec[pos_final] == empty:
            board_vec[pos_final] = board_vec[pos_init]
            board_vec[pos_init] = empty

            # Assign kings
            if pos_final in king_pos:
                board_vec[pos_final] = king_value

            # Remove eliminated pieces
            if move_type == 'jump':
                eliminated = int(jumps.iloc[pos_init, pos_final])
                print('Position eliminated: %d' % (eliminated + 1))
                assert board_vec[eliminated] == -chkr_value or -king_value
                board_vec[eliminated] = empty

            # Update the board
            board_vec = pd.DataFrame(np.reshape(board_vec, (8, 4)))
            self.state = board_vec
            return False

        else:
            return True


def play():

    # Alpha-numeric encoding of player turn: AI1 = 1, AI2 = -1
    turn = 1

    # Count number of invalid move attempts
    invalid_move_attempts = 0
    jumps_not_predicted = 0
    move_count = 1
    game_aborted = False

    # Initialize board object
    board = Board()

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

    # Define board entries and valid positions
    empty = 0
    black_chkr = 1
    black_king = 3
    black_king_pos = [28, 29, 30, 31]
    white_chkr = -black_chkr
    white_king = -black_king
    white_king_pos = [0, 1, 2, 3]
    valid_positions = list(range(32))
    odd_list = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    even_list = [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
    jumps = pd.read_csv(filepath_or_buffer='jumps.csv', header=-1, index_col=None)

    # Entries for neighbors are lists, with indices corresponding to direction as defined in parser_v7.py ...
    iv = ''
    neighbors = {0: [iv, 5, 4, iv],
                 1: [iv, 6, 5, iv],
                 2: [iv, 7, 6, iv],
                 3: [iv, iv, 7, iv],
                 4: [0, 8, iv, iv],
                 5: [1, 9, 8, 0],
                 6: [2, 10, 9, 1],
                 7: [3, 11, 10, 2],
                 8: [5, 13, 12, 4],
                 9: [6, 14, 13, 5],
                 10: [7, 15, 14, 6],
                 11: [iv, iv, 15, 7],
                 12: [8, 16, iv, iv],
                 13: [9, 17, 16, 8],
                 14: [10, 18, 17, 9],
                 15: [11, 19, 18, 10],
                 16: [13, 21, 20, 12],
                 17: [14, 22, 21, 13],
                 18: [15, 23, 22, 14],
                 19: [iv, iv, 23, 15],
                 20: [16, 24, iv, iv],
                 21: [17, 25, 24, 16],
                 22: [18, 26, 25, 17],
                 23: [19, 27, 26, 18],
                 24: [21, 29, 28, 20],
                 25: [22, 30, 29, 21],
                 26: [23, 31, 30, 22],
                 27: [iv, iv, 31, 23],
                 28: [24, iv, iv, iv],
                 29: [25, iv, iv, 24],
                 30: [26, iv, iv, 25],
                 31: [27, iv, iv, 26]
                 }

    next_neighbors = {0: [iv, 9, iv, iv],
                      1: [iv, 10, 8, iv],
                      2: [iv, 11, 9, iv],
                      3: [iv, iv, 10, iv],
                      4: [iv, 13, iv, iv],
                      5: [iv, 14, 12, iv],
                      6: [iv, 15, 13, iv],
                      7: [iv, iv, 14, iv],
                      8: [1, 17, iv, iv],
                      9: [2, 18, 16, 0],
                      10: [3, 19, 17, 1],
                      11: [iv, iv, 18, 2],
                      12: [5, 21, iv, iv],
                      13: [6, 22, 20, 4],
                      14: [7, 23, 21, 5],
                      15: [iv, iv, 22, 6],
                      16: [9, 25, iv, iv],
                      17: [10, 26, 24, 8],
                      18: [11, 27, 25, 9],
                      19: [iv, iv, 26, 10],
                      20: [13, 29, iv, iv],
                      21: [14, 30, 28, 12],
                      22: [15, 31, 29, 13],
                      23: [iv, iv, 30, 14],
                      24: [17, iv, iv, iv],
                      25: [18, iv, iv, 16],
                      26: [19, iv, iv, 17],
                      27: [iv, iv, iv, 18],
                      28: [21, iv, iv, iv],
                      29: [22, iv, iv, 20],
                      30: [23, iv, iv, 21],
                      31: [iv, iv, iv, 22]
                      }

    play()
