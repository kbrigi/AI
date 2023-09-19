import itertools
import random

import numpy as np

score = 0

def initGame():
    board = [[0] * 4 for _ in range(4)]
    addNewTile(board, 2)
    addNewTile(board, 2)
    return board

def addNewTile(board, type):
    freeTile = []
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                freeTile.append((i, j))
    if len(freeTile)-1 > 0:
        randTile = freeTile[np.random.randint(0, len(freeTile)-1)]
        if type == 2:
            board[randTile[0]][randTile[1]] = 2
        elif type == 24:
            board[randTile[0]][randTile[1]] = random.choice([2,4])
        
    return board

def checkMovesExist(board):
    # left/right
    for i in range(4):
        for j in range(3): 
            if board[i][j] == board[i][j+1]:
                return True
    
    # up/down
    for i in range(3):
        for j in range(4): 
            if board[i][j] == board[i+1][j]:
                return True
    
    return False

def checkGameStatus(board):
    if any(2048 in row for row in board):
        return 1
    elif not any(0 in row for row in board) and not checkMovesExist(board):
        return -1
    return 0

# matrix help functions
# -----------------------------------------------------
# stacks all tiles to the left <-
def stack(board):
    newBoard = [[0] * 4 for _ in range(4)]
    stacked = False
    for i in range(4):
        ii = 0
        for j in range(4):
            if board[i][j] != 0:
                newBoard[i][ii] = board[i][j]
                stacked = True
                ii+=1
    return newBoard, stacked

# combine all tiles mith the same value
def combine(board):
    score = 0
    combined = False
    for i in range(4):
        for j in range(3): 
            if board[i][j] != 0 and board[i][j] == board[i][j+1]:
                board[i][j] *= 2
                board[i][j+1] = 0
                score += board[i][j]
                combined = True
    return board, combined, score

def reverse(board):
    newBoard = []
    for i in range(4):
        newBoard.append([])
        for j in range(4):
            newBoard[i].append(board[i][3-j])
    return newBoard

def transpose(board):
    newBoard = [[0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4): 
            newBoard[i][j] = board[j][i]
    return newBoard
    

# matrix move functions
# -----------------------------------------------------
def move(board):
    board, stacked = stack(board)
    board, combined, score = combine(board)
    board, stacked = stack(board)
    moved = stacked or combined
    return board, moved, score

def left(board):
    # print("<-")
    board, moved, score = move(board)
    # board = addNewTile(board, 24)
    return board, moved, score

def right(board):
    # print("->")
    board = reverse(board)
    board, moved, score = move(board)
    board = reverse(board)
    # board = addNewTile(board, 24)
    return board, moved, score

def up(board):
    # print("↑")
    board = transpose(board)
    board, moved, score = move(board)
    board = transpose(board)
    # board = addNewTile(board, 24)
    return board, moved, score

def down(board):
    # print("↓")
    board = transpose(board)
    board = reverse(board)
    board, moved, score = move(board)
    board = reverse(board)
    board = transpose(board)
    # board = addNewTile(board, 24)
    return board, moved, score

def printBoard(board):
    for i in range(4):
        for j in range(4):
            print(board[i][j], end =" ")
        print('')
    print('')
    

def heuristic(matrix):
    weight=[[pow(4,6),pow(4,5),pow(4,4),pow(4,3)],
            [pow(4,5),pow(4,4),pow(4,3),pow(4,2)],
            [pow(4,4),pow(4,3),pow(4,2),pow(4,1)],
            [pow(4,3),pow(4,2),pow(4,1),pow(4,0)]]
    score=0
    for i in range(0,4):
        for j in range(0,4):
            score += +int(weight[i][j])*int(matrix[i][j])

    pen=0
    for i in range(0,4):
        for j in range(0,4):
            # not 1. row 
            if (i > 0):
                pen += abs(matrix[i][j] - matrix[i - 1][j])
            # 1. and 2. row
            if (i < 3):
                pen += abs(matrix[i][j] - matrix[i + 1][j])
            # not 1. column
            if (j > 0):
                pen += abs(matrix[i][j] - matrix[i][j - 1])
            # 1. and 2. col   
            if (j < 3):
                pen += abs(matrix[i][j] - matrix[i][j + 1])

    pen2=0  #for not empty tiles
    for i in range(0,4):
        for j in range(0,4):
            if(matrix[i][j]):
                pen2=pen2+1

    penalty = pen-2*pen2


    return score - penalty

def search(board, level, move):
    status = checkGameStatus(board)     #1 = won    -1 = lost
    score =  heuristic(board)

    if level == 0 or (move and (status == 1 or status == -1)) :
        return board, score

    # try all moves
    if move:
        all_moves = [left, up, down, right]
        for chosen_move in all_moves:
            child, _, _ = chosen_move(board)
            score = max(score, search(child, level-1, False)[1])
    # try all possible combinations of random 2/4
    else:
        score = 0
        zeros = [(i, j) for i, j in itertools.product(range(4), range(4)) if board[i][j] == 0]
        for i, j in zeros:
            board2 = board.copy()
            board2[i][j] = 2
            board4 = board.copy()
            board4[i][j] = 4

            score += (.9 * search(board2, level - 1, True)[1] / len(zeros) +
                        .1 * search(board4, level - 1, True)[1] / len(zeros))
    return board, score


def best_move(board):
    all_moves = [left, up, down, right]
    results = []
    for move in all_moves:
        board1, moved, _ = move(board)
        if moved and board != board1:
            result = search(board1, 4, False)
            results.append([result,board])
    
    return(max(results, key = lambda x: x[0][1])[0][0])

def play2048(board):
    won = checkGameStatus(board)
    moves = 0

    while won == 0:
        saved_borad = board.copy()
        moves += 1
        board = best_move(board)
        won = checkGameStatus(board)

        if won == 1:
            print("Congratulations! Moves: ", moves)
            printBoard(board)
            break
        elif won == -1 or saved_borad == board:
            printBoard(board)
            print("You lost! Moves:", moves)
            break
        board = addNewTile(board, 2)


board = initGame()
printBoard(board)
play2048(board)

