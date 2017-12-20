import numpy as np
import pandas as pd

import predict_move


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
# Board from black's prespective:
#   00  01  02  03
# 04  05  06  07
#   08  09  10  11
# 12  13  14  15
#   16  17  18  19
# 20  21  22  23
#   24  25  26  27
# 28  29  30  31
#
#
# Board from white's prespective:
#   31  30  29  28
# 27  26  25  24
#   23  22  21  20
# 19  18  17  16
#   15  14  13  12
# 11  10  09  08
#   07  06  05  04
# 03  02  01  00
#
#
# Label output:
#
# UR = Up Right
# DR = Down Right
# DL = Down Left
# UL = Up Left
#
#  UR DR DL UL
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0
#  0  0  0  0


# Entries for neighbors are lists, with indices corresponding to direction as
# defined above.
iv = ''
NEIGHBORS = {
    0: [iv, 5, 4, iv],
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
    31: [27, iv, iv, 26],
}

# The second neighbor in each direction.
NEXT_NEIGHBORS = {
    0: [iv, 9, iv, iv],
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
    31: [iv, iv, iv, 22],
}

# Constants for each space.
EMPTY = 0
BLACK_CHECKER = 1
BLACK_KING = 3
WHITE_CHECKER = -BLACK_CHECKER
WHITE_KING = -BLACK_KING

# Valid spaces to be kinged.
BLACK_KING_POS = [28, 29, 30, 31]
WHITE_KING_POS = [0, 1, 2, 3]

class Board(object):

    def __init__(self):
        self.state = pd.read_csv(filepath_or_buffer='board_init.csv', header=-1, index_col=None)
        self.invalid_attempts = 0

        # Constant, just in Board to avoid read_csv in global scope.
        self.JUMPS = pd.read_csv(filepath_or_buffer='jumps.csv', header=-1, index_col=None)

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
            king_value = BLACK_KING
            chkr_value = BLACK_CHECKER
            chkr_directions = [1, 2]
        else:
            king_value = WHITE_KING
            chkr_value = WHITE_CHECKER
            chkr_directions = [0, 3]

        board_state = self.state.copy()
        board_state = np.reshape(board_state.as_matrix(), (32,))

        for position in range(32):
            piece = board_state[position]
            neighbors_list = NEIGHBORS[position]
            next_neighbors_list = NEXT_NEIGHBORS[position]

            if piece == chkr_value:
                for direction in chkr_directions:
                    neighbor = neighbors_list[direction]
                    next_neighbor = next_neighbors_list[direction]
                    if neighbor == iv or next_neighbor == iv:
                        pass
                    elif board_state[next_neighbor] == EMPTY and (board_state[neighbor] == -chkr_value or board_state[neighbor] == -king_value):
                        valid_jumps.append([position, next_neighbor])

            elif piece == king_value:
                for direction in range(4):
                    neighbor = neighbors_list[direction]
                    next_neighbor = next_neighbors_list[direction]
                    if neighbor == iv or next_neighbor == iv:
                        pass
                    elif board_state[next_neighbor] == EMPTY and (board_state[neighbor] == -chkr_value or board_state[neighbor] == -king_value):
                        valid_jumps.append([position, next_neighbor])

        return valid_jumps

    def get_positions(self, move, player_type):

        # Extract starting position, and direction to move
        ind = np.argwhere(move == 1)[0]
        position = ind[0]
        direction = ind[1]

        jumps_available = self.find_jumps(player_type=player_type)

        neighbor = NEIGHBORS[position][direction]
        next_neighbor = NEXT_NEIGHBORS[position][direction]

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
            king_pos = BLACK_KING_POS
            king_value = BLACK_KING
            chkr_value = BLACK_CHECKER

        else:
            king_pos = WHITE_KING_POS
            king_value = WHITE_KING
            chkr_value = WHITE_CHECKER

        # print(pos_init, pos_final)
        board_vec = self.state.copy()
        board_vec = np.reshape(board_vec.as_matrix(), (32,))

        if (board_vec[pos_init] == chkr_value or board_vec[pos_init] == king_value) and board_vec[pos_final] == EMPTY:
            board_vec[pos_final] = board_vec[pos_init]
            board_vec[pos_init] = EMPTY

            # Assign kings
            if pos_final in king_pos:
                board_vec[pos_final] = king_value

            # Remove eliminated pieces
            if move_type == 'jump':
                eliminated = int(self.JUMPS.iloc[pos_init, pos_final])
                print('Position eliminated: %d' % (eliminated + 1))
                assert board_vec[eliminated] == -chkr_value or -king_value
                board_vec[eliminated] = EMPTY

            # Update the board
            board_vec = pd.DataFrame(np.reshape(board_vec, (8, 4)))
            self.state = board_vec
            return False

        else:
            return True
