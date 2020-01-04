import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from copy import deepcopy

class ConnectGame:
    def __init__(self, initial_state=None):
        self.board_dims = (6,7)
        self.rows, self.cols = self.board_dims
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = np.zeros(shape=self.board_dims)  # Counter positions for each player -1, 1
        self.col_tally = np.sum(np.abs(self.state), axis=0, dtype=int)  # No. pieces in each column
        self.check_end()  # Check game not done
        self.terminal, self.outcome = None, None

    def make_board(self):
        patches = []
        background = Polygon(((0, 0), (0, 7), (8, 7), (8, 0)), True)
        patches.append(background)
        patches += [Wedge((i, j), .3, 0, 360) for i in range(1, 8) for j in range(1, 7)]
        colors = [[1,1,1] for _ in range(len(patches))]
        colors[0] = [0.3,0.6,0.3]  # Green background
        p = PatchCollection(patches, facecolors=colors, alpha=0.5)
        return p

    def take_action(self, col_no, player_no):
        self.state[self.col_tally[col_no], col_no] = player_no
        self.col_tally = np.sum(np.abs(self.state), axis=0, dtype=int)
        assert 0 <= (len(np.where(self.state == -1)[0])-len(np.where(self.state == 1)[0])) <= 1  # 0,1 counter no. diff.
        return False

    def get_state(self):
        return self.state

    def get_turn(self):
        return 1 if np.sum(self.col_tally) % 2 else -1

    def legal_moves(self):
        return list(np.where(self.col_tally < 6)[0])

    def plot_board(self):
        rows1, cols1 = np.where(self.state == -1)
        rows2, cols2 = np.where(self.state == 1)
        ax = plt.gca()
        p = self.make_board()
        ax.add_collection(p)
        ax.set_xlim(0.5, 7+0.5)
        ax.set_yticklabels([])
        ax.set_ylim(0.5, 6+0.5)
        ax.scatter(cols1+1, rows1+1, marker='o', s=1000, c='#1f77b4')  # Player 1 blue
        ax.scatter(cols2+1, rows2+1, marker='o', s=1000, c='#ff7f0e')  # Player 2 orange
        plt.show(block=False)

    def check_win(self):
        sum_filter = np.ones(4)
        vwin = np.min(np.apply_along_axis(lambda m: np.convolve(m, sum_filter, 'same'), axis=0, arr=self.state))
        hwin = np.min(np.apply_along_axis(lambda m: np.convolve(m, sum_filter, 'same'), axis=1, arr=self.state))
        vwin2 = np.max(np.apply_along_axis(lambda m: np.convolve(m, sum_filter, 'same'), axis=0, arr=self.state))
        hwin2 = np.max(np.apply_along_axis(lambda m: np.convolve(m, sum_filter, 'same'), axis=1, arr=self.state))
        if np.min([hwin, vwin]) == -4:
            return True, -1
        elif np.max([hwin2, vwin2]) == 4:
            return True, 1
        for i in range(-2, 4):  # Possible winning diagonals
            diag_arr = np.diagonal(self.state, offset=i)
            antidiag_arr = np.flipud(self.state).diagonal(offset=i)
            diags = np.apply_along_axis(lambda m: np.convolve(m, sum_filter, 'same'), axis=0, arr=diag_arr)
            anti_diags = np.apply_along_axis(lambda m: np.convolve(m, sum_filter, 'same'), axis=0, arr=antidiag_arr)
            if np.min(np.array([diags,anti_diags])) == -4:
                return True, -1
            elif np.max(np.array([diags,anti_diags])) == 4:
                return True, 1
        return False, 0

    def check_end(self):
        winner, player_no = self.check_win()
        if winner:
            self.terminal = True
            self.outcome = player_no
            return True
        if not self.legal_moves():
            self.terminal = True
            self.outcome = 0  # Draw
            return True
        return False

    def clone(self):
        """Return copy of board for simulations."""
        return deepcopy(self)

if __name__ == "__main__":
    cg = ConnectGame()
    cg.take_action(3,-1)
    cg.take_action(4,1)
    cg.take_action(4, -1)

    """
    cg.take_action(0,-1)
    cg.take_action(0,1)
    cg.take_action(1,-1)
    cg.take_action(1,1)
    cg.take_action(1,-1)
    cg.take_action(2,-1)
    cg.take_action(2,1)
    cg.take_action(3,-1)
    cg.take_action(2,1)
    cg.take_action(3,-1)
    cg.take_action(3,1)
    cg.take_action(3,-1)"""

    cg.plot_board()
    print(cg.col_tally)
    from timeit import default_timer as timer
    start = timer()
    for i in range(10):
        cg.check_win()
    end = timer()
    print(end-start)
    print(cg.legal_moves())
    if not cg.legal_moves():
        print('hey')