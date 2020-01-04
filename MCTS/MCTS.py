import numpy as np
from Connect4 import ConnectGame
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from copy import deepcopy

class Node:
    def __init__(self, game, parent=None, action=None):
        game_copy = game.clone()
        self.c = 1
        self.parent = parent
        if self.parent:
            self.tree_level = self.parent.tree_level+1
        else:
            self.tree_level = 0  # Nodes level on tree
        self.action = action  # Action leading to node
        self.children = {}
        self.state = game_copy.get_state()
        self.unexplored_moves = game_copy.legal_moves()
        self.leaf = False
        if not self.unexplored_moves:
            self.leaf = True
        self.pts = 0
        self.games = 0
        self.turn = game_copy.get_turn()

    def update(self, outcome):
        if outcome == 0:
            self.pts += 0.5
        elif outcome == -self.turn:  # Node is better to choose if other player loses.
            self.pts += 1
        self.games += 1

    def expand(self, game, action):
        """Expand search tree at node"""
        child = Node(game.clone(), self, action)
        self.children[action] = child
        self.unexplored_moves.remove(action)
        return child

    def get_best_child(self):
        sort_key = lambda x: x.pts/x.games + self.c*np.sqrt(2 * np.log(self.games) / x.games)  # Metric to sort children
        return sorted(self.children.values(), key=sort_key)[-1]  # Return child with largest UCT value


class MCTSagent:
    def __init__(self, game):
        self.game = game
        self.simulation_game = self.game.clone()
        self.root_node = Node(self.game)
        self.nodes = dict()  # Store all nodes
        self.no_nodes = 0

    def move_root(self, action):
        if action not in self.root_node.children:
            self.root_node = Node(self.game)
        else:
            self.root_node = self.root_node.children[action]  # Shift root to chosen child
            self.root_node.parent = None
        print(f"Moved to level {self.root_node.tree_level} of search tree")

    def _select(self, node):
        # Selection step, traverse tree to an unexpanded node
        while node.unexplored_moves == [] and node.children != {}:
            child_node = node.get_best_child()
            self.simulation_game.take_action(child_node.action, node.turn)  # Take action to get to next node
            node = child_node
        return node

    def _expand(self, node):
        """Expand current node."""
        if node.unexplored_moves != []:
            move = random.choice(node.unexplored_moves)
            self.simulation_game.take_action(move, node.turn)
            node = node.expand(deepcopy(self.simulation_game), move)
        self.no_nodes += 1
        return node

    def _simulate(self, node):
        current_turn = node.turn
        while not self.simulation_game.check_end():
            move = random.choice(self.simulation_game.legal_moves())
            self.simulation_game.take_action(move, current_turn)
            current_turn *= -1
        return True

    def _update(self, node):
        while node is not None:
            node.update(self.simulation_game.outcome)
            node = node.parent
        return True

    def run(self, iterations=500):
        """Run MCTS algorithm"""
        for i in range(iterations):
            node = self.root_node  # Revert back to root node at every iteration
            self.simulation_game = self.game.clone()  # Take copy of game at current state
            assert (self.root_node.state == self.simulation_game.state).all()
            node = self._select(node)
            node = self._expand(node)
            assert self._simulate(node)
            assert self._update(node)
        sort_key = lambda x: x.pts/x.games  # Choose best possible action in real game
        best_child = sorted(self.root_node.children.values(), key=sort_key)[-1]
        return self.root_node.children, best_child  # Return child with largest win %

if __name__ == "__main__":
    cg = ConnectGame()
    plt.ion()
    cg.plot_board()
    root_node = None  # No node to begin with
    user = random.choice([-1, 1])
    cpu = 1 if user == -1 else -1
    mcts = MCTSagent(cg)
    initial_iterations, iterations = 1000, 250
    children, best_child = mcts.run(initial_iterations)
    if cpu == -1:  # Computer goes first
        cg.take_action(best_child.action, cpu)
        cg.plot_board()
        mcts.move_root(best_child.action)  # Move root node down tree

    while not cg.check_end():  # While game not finished
        valid_action = False
        while True:
            try:
                user_action = int(input("Choose a column (1-7) to drop counter: "))-1
                assert (0 <= user_action <= 6)
                cg.take_action(user_action, user)
                break
            except:
                print("Invalid input given, please try again")
                pass
        cg.plot_board()
        if cg.check_end():
            break
        mcts.move_root(user_action)  # Move root node down tree

        print("Computer thinking...", end=' ')
        start = timer()
        children, best_child = mcts.run(iterations)
        end = timer()
        print(f"Thinking time: {round(end-start,3)} seconds")
        print("Estimated win percentages:")
        action_ratings = [(child.action+1, round(100*child.pts/child.games,2)) for child in children.values()]
        print(sorted(action_ratings, key=lambda x: x[0]))
        print(f"Action chosen: {best_child.action+1}")
        cg.take_action(best_child.action, cpu)
        cg.plot_board()
        mcts.move_root(best_child.action)  # Move root node down tree
    if cg.outcome == user:
        print("Human wins")
    elif cg.outcome == cpu:
        print("Computer wins")
    else:
        print("Game drawn")