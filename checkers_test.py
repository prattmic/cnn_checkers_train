import logging
logging.getLogger('tensorflow').disabled = True

import numpy as np
import unittest

import checkers

class TestBoard(unittest.TestCase):

    def test_update(self):
        #   x   x   x   x
        # x   x   x   x
        #   x   x   x   x
        # -   -   -   -
        #   -   _   -   -
        # o   o   o   o
        #   o   o   o   o
        # o   o   o   o
        #
        #   00  01  02  03
        # 04  05  06  07
        #   08  09  10  11
        # 12  13  14  15
        #   16  17  18  19
        # 20  21  22  23
        #   24  25  26  27
        # 28  29  30  31
        #
        # x = black
        # o = white

        board = checkers.Board()

        # Valid moves by black.
        self.assertFalse(board.copy().update([8, 12], 'black'))
        self.assertFalse(board.copy().update([8, 13], 'black'))

        # Invalid moves by black.

        # Move too far.
        self.assertTrue(board.copy().update([8, 17], 'black'))

        # Move non-diagonal.
        self.assertTrue(board.copy().update([8, 16], 'black'))

        # Jump over own piece.
        self.assertTrue(board.copy().update([5, 12], 'black'))


        # Valid moves by white.
        self.assertFalse(board.copy().update([22, 18], 'white'))
        self.assertFalse(board.copy().update([22, 17], 'white'))

        # Invalid moves by white.

        # Move too far.
        self.assertTrue(board.copy().update([22, 13], 'white'))

        # Move non-diagonal.
        self.assertTrue(board.copy().update([22, 14], 'white'))

        # Jump over own piece.
        self.assertTrue(board.copy().update([27, 18], 'black'))


        b = checkers.BLACK_CHECKER
        w = checkers.WHITE_CHECKER
        e = checkers.EMPTY

        board.state = np.array([
            [e, e, e, e],
            [e, e, e, e],
            [b, b, e, e],
            [e, w, e, e],
            [b, w, e, e],
            [e, e, e, e],
            [e, e, e, e],
            [e, e, e, e],
        ])
        #   _   _   _   _
        # _   _   _   _
        #   x   x   _   _
        # -   o   -   -
        #   x   o   -   -
        # _   _   _   _
        #   _   _   _   _
        # _   _   _   _
        #
        #   00  01  02  03
        # 04  05  06  07
        #   08  09  10  11
        # 12  13  14  15
        #   16  17  18  19
        # 20  21  22  23
        #   24  25  26  27
        # 28  29  30  31

        # Can't move backwards.
        self.assertTrue(board.copy().update([17, 22], 'white'))

        # Valid jump.
        self.assertFalse(board.copy().update([13, 6], 'white'))

        # Can't jump backwards.
        self.assertTrue(board.copy().update([13, 20], 'white'))

        # Can't end jump on non-empty space.
        self.assertTrue(board.copy().update([9, 16], 'black'))


        B = checkers.BLACK_KING
        W = checkers.WHITE_KING

        board.state = np.array([
            [e, e, e, e],
            [e, e, e, e],
            [B, b, e, e],
            [e, W, e, e],
            [b, w, e, e],
            [e, e, e, e],
            [e, e, e, e],
            [e, e, e, e],
        ])
        #   _   _   _   _
        # _   _   _   _
        #   X   x   _   _
        # -   O   -   -
        #   x   o   -   -
        # _   _   _   _
        #   _   _   _   _
        # _   _   _   _
        #
        #   00  01  02  03
        # 04  05  06  07
        #   08  09  10  11
        # 12  13  14  15
        #   16  17  18  19
        # 20  21  22  23
        #   24  25  26  27
        # 28  29  30  31

        # King can move backwards.
        self.assertFalse(board.copy().update([8, 5], 'black'))

        # King can jump backwards.
        self.assertFalse(board.copy().update([13, 20], 'white'))
