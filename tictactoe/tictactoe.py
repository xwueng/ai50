"""
Tic Tac Toe Player
"""

import math
import copy
from collections import Counter

X = "X"
O = "O"
EMPTY = None


def cols_to_rows(board):
    fbd = []
    for c in range(3):
        row = []
        for r in range(3):
            row.append(board[r][c])
        fbd.append(row)
    return fbd


def diagnos_to_rows(board):
    tempbd = []
    fwdrow = []
    bwdrow = []
    for c in range(3):
        for r in range(3):
            if c == r:
                bwdrow.append(board[r][c])
                if c == 1:
                    fwdrow.append(board[r][c])
            elif c + r == 2:
                fwdrow.append(board[r][c])
    tempbd.append(bwdrow)
    tempbd.append(fwdrow)
    return tempbd


def search_winner(rect):
    winner = None
    if [X] * 3 in rect:
        winner = X
    elif [O] * 3 in rect:
        winner = O
    return winner


def initial_state():
    """
    Returns starting board of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    flatbd = [item for row in board for item in row]
    counts = Counter(flatbd)
    xcount = counts[X]
    ocount = counts[O]
    if xcount == ocount:
        return X
    elif xcount == ocount + 1:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                empty_cells.append((i, j))
    # print(f"--actions() board {board}, empty_cells:{empty_cells}")
    return empty_cells
    # raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    expected_board = copy.deepcopy(board)
    # print(f"--result board {board}, action:{action}")
    if len(action) < 2:
        raise Exception(f"result()  action len < 2: {action}, board: {board}")
    else:
        x = action[0]
        y = action[1]

    if expected_board[x][y] == EMPTY:
        expected_board[x][y] = player(expected_board)
        return expected_board
    else:
        raise Exception(f"Sorry, you can't play in cell {action} because \
        it's taken by {expected_board[x][y]}")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    #search XXX/OOO in rows
    winner = None
    winner = search_winner(board)

    # search XXX/OOO in cols
    if winner == None:
        flatbd = cols_to_rows(board)
        winner = search_winner(flatbd)

    # search XXX/OOO diagnolly
    if winner == None:
        flatbd = diagnos_to_rows(board)
        winner = search_winner(flatbd)

    return winner


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    game_over = winner(board) != None
    if not game_over:
        flatbd = [item for row in board for item in row]
        counts = Counter(flatbd)
        game_over = counts[EMPTY] == 0
    return game_over


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_dict = {'X': 1, 'O': -1, None: 0}
    game_winner = winner_dict[winner(board)]
    return game_winner


def min_value(board):
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        # print(f"--max_value() board {board}, action:{action}")
        if type(action) != tuple:
            raise Exception(f"min_value action len < 2: {action}, board: {board}")
        else:
            # print(f"in min_value() result(board, action: {result(board, action)}")
            v = min(v, max_value(result(board, action)))
            vaction = action
            # print(f"min_value() v: {v}, vaction: {vaction}")
    return v


def max_value(board):
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        # print(f"--max_value() board {board}, action:{action}")
        if type(action) != tuple:
            raise Exception(f"max_value action len < 2: {action}, board: {board}")
        else:
            # print(f"in max_value() result(board, action): {result(board, action)}")
            v = max(v, min_value(result(board, action)))
            vaction = action
            # print(f"max_value() v: {v}, vaction: {vaction}")
    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    print(f"\n-------- in minmax board {board}")
    if player(board) == X:
        type = 'max'
    else:
        type = 'min'

    if terminal(board):
        return utility(board)
    if type == 'max':
        v = -math.inf
    else:
        v = math.inf
    vgoal = v
    goal_action = ()
    for action in actions(board):
        # print(f"\n-------- minmax {type} board {board}, action:{action}")
        # print(f"\n----- max_value() result(board, action): {result(board, action)}")
        if type == 'max':
            v = max(v, min_value(result(board, action)))
            update_vgoal = v > vgoal
        else:
            v = min(v, max_value(result(board, action)))
            update_vgoal = v < vgoal

        if update_vgoal:
            vgoal = v
            goal_action = action
            # print(f"max_value() v: {v}, vgoal: {vgoal}, goal_action: {goal_action}")
    return goal_action
