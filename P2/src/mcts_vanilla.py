
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 1000
exploration_factor = 2.

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    while not board.is_ended:
        if not node.child_nodes:
            return expand_leaf(node, board, state)
        else:
            # generates a list of tuples (ucb, action) for every untried action
            ucbs = [(ucb(node.child_nodes[action], bot_identity == board.current_player), action) for action in node.untried_actions]
            
            # maximizes ucb if its the player's turn otherwise minimizes it if it's the opponent's turn
            best_action = max(ucbs)[1] if bot_identity == board.current_player else min(ucbs)[1]
            
            # updates new node to be best child and state to be next state
            node = node.child_nodes[best_action]
            state = board.next_state(state, best_action)
    
    return node, state

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    if board.is_ended(state):
        return False
    
    action = choice(node.untried_actions)
    new_state = board.next_state(action)
    child = MCTSNode(parent=node, parent_action=action, action_list=board.legal_actions(new_state))
    node.child_nodes[action] = child
    
    return child, new_state


def rollout(board: Board, state): #simulation stage of MCTS
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    while not board.is_ended:
        state = board.next_state(state, choice(board.legal_actions(state)))

    return state

def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    if node:
        node.wins += 1 if won else -1
        node.visits += 1
        backpropagate(node.parent, not won)

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    
    max_win_ratio = 0
    # doing the node stuff first to reduce computations
    # c * sqrt(ln(t))
    exploration_estimate_top = exploration_factor * sqrt(log(node.visits))
    for child in node.child_nodes.values():
        # w/v
        child_win_ratio = child.wins / child.visits
        
        # sqrt(v)
        exploration_estimate_bottom = sqrt(child.visits)
        
        # w/v + c * (sqrt(ln(t)) / sqrt(v))
        win_ratio = child_win_ratio + exploration_estimate_top / exploration_estimate_bottom
        
        # maximize this ratio
        if win_ratio > max_win_ratio:
            max_win_ratio = win_ratio
           
    return max_win_ratio if is_opponent else 1 - max_win_ratio

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    
    max_ucb = 0
    best_action = None
    for action in root_node.untried_actions:
        child = root_node.child_nodes[action]
        child_ucb = ucb(child, True)
        if child_ucb > max_ucb:
            best_action = action
            max_ucb = child_ucb
    
    return best_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node

        # Do MCTS - This is all you!
        # ...
        node = traverse_nodes(root_node)
        state = rollout(board, state)
        sim = is_win(board, state, bot_identity)
        backpropagate(node, sim)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)
    
    print(f"Action chosen: {best_action}")
    return best_action