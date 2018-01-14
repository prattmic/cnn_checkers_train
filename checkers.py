# Original author: Chris Larson
# All Rights Reserved (2016)


import numpy as np

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


# Invalid neighbor
INVALID = ''

# Entries for neighbors are lists, with indices corresponding to direction as
# defined above.
iv = INVALID  # shorthand
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


def build_jumps():
    """Creates the jump table."""
    jumps = np.full((32, 32), -1)

    # Valid jumps.
    jumps[0, 9] = 5
    jumps[1, 8] = 5
    jumps[1, 10] = 6
    jumps[2, 9] = 6
    jumps[2, 11] = 7
    jumps[3, 10] = 7
    jumps[4, 13] = 8
    jumps[5, 12] = 8
    jumps[5, 14] = 9
    jumps[6, 13] = 9
    jumps[6, 15] = 10
    jumps[7, 14] = 10
    jumps[8, 1] = 5
    jumps[8, 17] = 13
    jumps[9, 0] = 5
    jumps[9, 2] = 6
    jumps[9, 16] = 13
    jumps[9, 18] = 14
    jumps[10, 1] = 6
    jumps[10, 3] = 7
    jumps[10, 17] = 14
    jumps[10, 19] = 15
    jumps[11, 2] = 7
    jumps[11, 18] = 15
    jumps[12, 5] = 8
    jumps[12, 21] = 16
    jumps[13, 4] = 8
    jumps[13, 6] = 9
    jumps[13, 20] = 16
    jumps[13, 22] = 17
    jumps[14, 5] = 9
    jumps[14, 7] = 10
    jumps[14, 21] = 17
    jumps[14, 23] = 18
    jumps[15, 6] = 10
    jumps[15, 22] = 18
    jumps[16, 9] = 13
    jumps[16, 25] = 21
    jumps[17, 8] = 13
    jumps[17, 10] = 14
    jumps[17, 24] = 21
    jumps[17, 26] = 22
    jumps[18, 9] = 14
    jumps[18, 11] = 15
    jumps[18, 25] = 22
    jumps[18, 27] = 23
    jumps[19, 10] = 15
    jumps[19, 26] = 23
    jumps[20, 13] = 16
    jumps[20, 29] = 24
    jumps[21, 12] = 16
    jumps[21, 14] = 17
    jumps[21, 28] = 24
    jumps[21, 30] = 25
    jumps[22, 13] = 17
    jumps[22, 15] = 18
    jumps[22, 29] = 25
    jumps[22, 31] = 26
    jumps[23, 14] = 18
    jumps[23, 30] = 26
    jumps[24, 17] = 21
    jumps[25, 16] = 21
    jumps[25, 18] = 22
    jumps[26, 17] = 22
    jumps[26, 19] = 23
    jumps[27, 18] = 23
    jumps[28, 21] = 24
    jumps[29, 20] = 24
    jumps[29, 22] = 25
    jumps[30, 21] = 25
    jumps[30, 23] = 26
    jumps[31, 22] = 26

    return jumps


# JUMPS[from, to] returns the position jumped over.
JUMPS = build_jumps()


def new_state():
    # P1 is white, P2 is black
    # TODO(prattmic): swap this?
    #
    # axis 0 - state history
    # axis 1 - 0 P1 normal, P1 king, P2 normal, P2 king, color to make move
    # axis 2 - Board y axis
    # axis 3 - Board x axis
    return np.zeros((2, 5, 8, 4), np.int8)


class State(object):
    """Helper wrapper around checkers state numpy array.

    This object simply provides convience methods. It contains no state beyond
    the state array itself.
    """

    def __init__(self, array=None):
        self.array = array
        if self.array is None:
            self.array = new_state()

    @staticmethod
    def unmerge(merged, player):
        """Returns State from a single merged 8x4 2-D matrix of board state.

        Args:
            merged: 8x4 2-D matric of board state.
            player: Player to make next move (P1 = 1, P2 = -1)
        """
        a = new_state()

        np.place(a[0, 0], merged == WHITE_CHECKER, 1)
        np.place(a[0, 1], merged == WHITE_KING, 1)
        np.place(a[0, 2], merged == BLACK_CHECKER, 1)
        np.place(a[0, 3], merged == BLACK_KING, 1)
        a[0, 4] = np.full_like(a[0, 4], player)

        return State(a)

    def merged(self):
        """Returns a single 8x4 2-D matrix of board state.

        Uses EMPTY, BLACK_CHECKER, BLACK_KING, WHITE_CHECKER, WHITE_KING values.
        """
        # Only use the top move.
        a = self.array[0].copy()

        white_checker = a[0]
        np.place(white_checker, white_checker>0, WHITE_CHECKER)

        white_king = a[1]
        np.place(white_king, white_king>0, WHITE_KING)

        black_checker = a[2]
        np.place(black_checker, black_checker>0, BLACK_CHECKER)

        black_king = a[3]
        np.place(black_king, black_king>0, BLACK_KING)

        return white_checker + white_king + black_checker + black_king

    def merged_vec(self):
        """Returns a 32-entry 1-D vector of board state.

        Uses EMPTY, BLACK_CHECKER, BLACK_KING, WHITE_CHECKER, WHITE_KING values.
        """
        return self.merged().reshape((32,))

    def copy(self):
        return State(self.array.copy())

    def history(self, n):
        """Pops the n newest moves off the state."""
        # Delete along history axis (0).
        a = self.array
        for i in range(n):
            a = np.delete(a, 0, 0)
        return State(a)

    def color(self):
        """Returns the color to make the next move."""
        # Grab the first element in the color plane. The entire plane is
        # identical.
        return self.array[0, 4, 0, 0]


def build_init():
    b = BLACK_CHECKER
    w = WHITE_CHECKER
    e = EMPTY

    a = np.array([
        [b, b, b, b],
        [b, b, b, b],
        [b, b, b, b],
        [e, e, e, e],
        [e, e, e, e],
        [w, w, w, w],
        [w, w, w, w],
        [w, w, w, w],
    ])

    # Player 1's turn.
    s = State.unmerge(a, 1)
    s.array.flags.writeable = False

    return s


# Initial board state. Copy before modifying.
BOARD_INIT = build_init()


class Board(object):

    def __init__(self, state=None):
        self.state = state
        if self.state is None:
            self.state = BOARD_INIT.copy()

        self.jumps_not_predicted = 0
        self.invalid_move_attempts = 0

    def copy(self):
        """Returns a deep copy of this board."""
        b = Board()
        b.state = self.state.copy()
        b.jumps_not_predicted = self.jumps_not_predicted
        b.invalid_move_attempts = self.invalid_move_attempts
        return b

    def board_state(self, player_type):
        if player_type == 'white':
            return -self.state[::-1, ::-1]
        elif player_type == 'black':
            return self.state

    def print_board(self):
        for i in range(8):
            for j in range(4):
                if i % 2 == 0:
                    print(' ', end=' ')

                # We print the board 'upside down'. i.e., black/position 0 at
                # the bottom.
                v = self.state[7 - i, 3 - j]

                if v == BLACK_CHECKER:
                    print('x', end=' ')
                elif v == BLACK_KING:
                    print('X', end=' ')
                elif v == EMPTY:
                    print('-', end=' ')
                elif v == WHITE_CHECKER:
                    print('o', end=' ')
                else:
                    print('O', end=' ')

                if i % 2 != 0:
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

        board_state = np.reshape(self.state, (32,))

        for position in range(32):
            piece = board_state[position]
            neighbors_list = NEIGHBORS[position]
            next_neighbors_list = NEXT_NEIGHBORS[position]

            if piece == chkr_value:
                for direction in chkr_directions:
                    neighbor = neighbors_list[direction]
                    next_neighbor = next_neighbors_list[direction]
                    if neighbor == INVALID or next_neighbor == INVALID:
                        pass
                    elif board_state[next_neighbor] == EMPTY and (board_state[neighbor] == -chkr_value or board_state[neighbor] == -king_value):
                        valid_jumps.append([position, next_neighbor])

            elif piece == king_value:
                for direction in range(4):
                    neighbor = neighbors_list[direction]
                    next_neighbor = next_neighbors_list[direction]
                    if neighbor == INVALID or next_neighbor == INVALID:
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

    def generate_move(self, player_type, output_type, predictor):

        board_state = self.board_state(player_type=player_type)
        moves_list = list()

        # Assumes both AI are the same CNN model.
        moves, probs = predict_move.predict_cnn(board_state, output=output_type, predictor=predictor)

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

    def valid_moves(self, player_type):
        jumps = self.find_jumps(player_type)

        if player_type == 'black':
            king_value = BLACK_KING
            chkr_value = BLACK_CHECKER
            chkr_directions = [1, 2]
        else:
            king_value = WHITE_KING
            chkr_value = WHITE_CHECKER
            chkr_directions = [0, 3]

        board_state = np.reshape(self.state, (32,))
        moves = []

        for position in range(32):
            piece = board_state[position]
            neighbors_list = NEIGHBORS[position]

            if piece == chkr_value:
                directions = chkr_directions
            elif piece == king_value:
                directions = range(4)
            else:
                # Not our piece.
                continue

            found_jump = False
            for jump in jumps:
                if jump[0] == position:
                    found_jump = True
                    moves.append(jump)

            if found_jump:
                # If a jump is available, it must be taken.
                continue

            for direction in directions:
                neighbor = neighbors_list[direction]
                if neighbor == INVALID:
                    continue

                if board_state[neighbor] == EMPTY:
                    moves.append([position, neighbor])

        return moves

    def update(self, positions, player_type):
        # Extract the initial and final positions into ints.
        [pos_init, pos_final] = int(positions[0]), (positions[1])

        if player_type == 'black':
            king_pos = BLACK_KING_POS
            king_value = BLACK_KING
            chkr_value = BLACK_CHECKER
            chkr_directions = [1, 2]
        else:
            king_pos = WHITE_KING_POS
            king_value = WHITE_KING
            chkr_value = WHITE_CHECKER
            chkr_directions = [0, 3]

        board_vec = np.reshape(self.state, (32,))

        # 1. Move must start from one of our pieces.
        if board_vec[pos_init] != chkr_value and board_vec[pos_init] != king_value:
            print("Invalid move %s: doesn't start from our piece" % (positions,))
            return True

        # 2. Move must end on an empty space.
        if board_vec[pos_final] != EMPTY:
            print("Invalid move %s: doesn't end on empty piece" % (positions,))
            return True

        # 3. Move must go to a neighbor or be a valid jump.
        jumped = None

        neighbors = NEIGHBORS[pos_init]
        next_neighbors = NEXT_NEIGHBORS[pos_init]
        if board_vec[pos_init] != king_value:
            # Normal checkers can only go in two directions.
            neighbors = [neighbors[chkr_directions[0]], neighbors[chkr_directions[1]]]
            next_neighbors = [next_neighbors[chkr_directions[0]], next_neighbors[chkr_directions[1]]]

        if pos_final in neighbors:
            # OK.
            pass
        else:
            # 3a. only allow forward jumps for non-kings.
            if pos_final not in next_neighbors:
                print("Invalid move %s: not neighbor or jump" % (positions,))
                return True

            jumped = JUMPS[pos_init, pos_final]
            # Jump must be valid if in was in next_neighbors.
            assert jumped != -1

            # 3b. Jump must jump opposing piece.
            if board_vec[jumped] != -chkr_value and board_vec[jumped] != -king_value:
                print("Invalid move %s: not jumping opposing piece" % (positions,))
                return True

        # This move is valid. Actually perform update.
        board_vec[pos_final] = board_vec[pos_init]
        board_vec[pos_init] = EMPTY

        # Assign kings.
        if pos_final in king_pos:
            board_vec[pos_final] = king_value

        # Remove eliminated pieces.
        if jumped is not None:
            print('Position eliminated: %d' % jumped)
            board_vec[jumped] = EMPTY

        # Update the board.
        self.state = np.reshape(board_vec, (8, 4))
        return False

    def move_ai(self, player_type, predictor):
        """Automatically complete a move as the AI.

        Returns True if a move was made successfully, False to abort the game.
        """
        # Call model to generate move
        moves_list, probs = self.generate_move(player_type=player_type, output_type='top-10', predictor=predictor)
        print(np.array(moves_list))
        print(probs)

        # Check for available jumps, cross check with moves
        available_jumps = self.find_jumps(player_type=player_type)

        first_move = True

        # Handles situation in which jump is available
        if len(available_jumps) > 0:
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
                        self.jumps_not_predicted += 1

                    initial_position = available_jumps[0][0]
                    if not (first_move or final_position == initial_position):
                        break
                    final_position = available_jumps[0][1]
                    initial_piece = np.reshape(self.state, (32,))[initial_position]
                    move_illegal = self.update(available_jumps[0], player_type=player_type)

                    if move_illegal:
                        print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[0]) + 1))
                        return False
                    else:
                        print("%s move: %s" % (player_type, (np.array(available_jumps[0]) + 1)))
                        available_jumps = self.find_jumps(player_type=player_type)
                        final_piece = np.reshape(self.state, (32,))[final_position]
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
                            initial_piece = np.reshape(self.state, (32,))[initial_position]
                            move_illegal = self.update(move, player_type=player_type)

                            if move_illegal:
                                print('Model and Find jumps function predicted an invalid move: %s' % (np.array(move) + 1))
                            else:
                                print("%s move: %s" % (player_type, (np.array(move) + 1)))
                                move_predicted = True
                                available_jumps = self.find_jumps(player_type=player_type)
                                final_piece = np.reshape(self.state, (32,))[final_position]
                                if len(available_jumps) == 0 or final_piece != initial_piece:
                                    jump_available = False
                                break

                    if not move_predicted:
                        print('Model did not output any of the available jumps. Move picked randomly among valid options.')
                        self.jumps_not_predicted += 1
                        ind = np.random.randint(0, len(available_jumps))

                        initial_position = available_jumps[ind][0]
                        if not (first_move or final_position == initial_position):
                            break
                        final_position = available_jumps[ind][1]
                        initial_piece = np.reshape(self.state, (32,))[initial_position]
                        move_illegal = self.update(available_jumps[ind], player_type=player_type)

                        if move_illegal:
                            print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[ind]) + 1))
                            return False
                        else:
                            available_jumps = self.find_jumps(player_type=player_type)
                            final_piece = np.reshape(self.state, (32,))[final_position]
                            if len(available_jumps) == 0 or final_piece != initial_piece:
                                jump_available = False

                first_move = False

        # For standard moves
        else:
            move_illegal = True
            while move_illegal:

                count = 1
                for move in moves_list:

                    move_illegal = self.update(move, player_type=player_type)

                    if move_illegal:
                        print('model predicted invalid move (%s)' % (np.array(move) + 1))
                        print(probs[count - 1])
                        self.invalid_move_attempts += 1
                        count += 1
                    else:
                        print('%s move: %s' % (player_type, (np.array(move) + 1)))
                        break

                if move_illegal:
                    print("The model failed to provide a valid move. Game aborted.")
                    print(np.array(moves_list) + 1)
                    print(probs)
                    return False

        return True

def play(player1_move, player2_move):
    """Play a full game of checkers.

    player1_move and player2_move are functions that perform a move for that
    player. They are passed the board and should update the board with their
    move then return True. If they return False the game is aborted.
    """

    # Alpha-numeric encoding of player turn: Player 1 = 1, Player 2 = -1
    turn = -1

    move_count = 1

    # Initialize board object
    board = Board()

    print('Player 1 is playing white, Player 2 is playing black.')
    print('There is no GUI for this game. Feel free to run an external program in 2-player mode alongside this game.')
    print('\n')

    while True:

        print('\n' * 2)
        print('=======================================================')

        abort = False

        # White turn
        if turn == 1:
            print('Move %d: white' % move_count)
            board.print_board()
            abort = not player1_move(board)

        # Black turn
        else:
            print('Move %d: black' % move_count)
            board.print_board()
            abort = not player2_move(board)

        # Check game status
        num_black_pieces = len(np.argwhere(board.state > 0))
        num_white_pieces = len(np.argwhere(board.state < 0))

        if num_black_pieces == 0:
            winner = 'white'
            break
        elif num_white_pieces == 0:
            winner = 'black'
            break
        elif move_count >= 100:
            winner = 'draw'
            break
        elif abort:
            winner = 'n/a'
            break

        move_count += 1
        turn *= -1

    # Print out game stats
    end_board = board.state
    print('Ending board:')
    print(board.board_state(player_type='white'))
    num_black_chkr = len(np.argwhere(end_board == BLACK_CHECKER))
    num_black_king = len(np.argwhere(end_board == BLACK_KING))
    num_white_chkr = len(np.argwhere(end_board == WHITE_CHECKER))
    num_white_king = len(np.argwhere(end_board == WHITE_KING))

    if winner == 'draw':
        print('The game ended in a draw.')
    else:
        print('%s wins' % winner)

    print('Total number of moves: %d' % move_count)
    print('Remaining white pieces: (checkers: %d, kings: %d)' % (num_white_chkr, num_white_king))
    print('Remaining black pieces: (checkers: %d, kings: %d)' % (num_black_chkr, num_black_king))
    print('Invalid move attempts: %d' % board.invalid_move_attempts)
    print('Jumps not predicted: %d' % board.jumps_not_predicted)
