"""
Tic Tac Toe Player
"""
import math
import sys
import time

from datetime import datetime
from termcolor import colored
from copy import deepcopy


X = "X"
O = "O"
EMPTY = None



# Error: Action not Possible
class ActionNotPossibleError(Exception):
    def __init__(self, expression, message):
        self.message = "Action {} is not possible!"
        print(self.message.format(str(expression)))
    
    

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY ,EMPTY],
            [EMPTY, EMPTY, EMPTY]]
    

def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # If initial state, first move goes to X
    if board == initial_state(): 
        return X
   
    # If game is over, don't return player
    if terminal(board): 
        return "Game Over"
    
    # Else count each player's moves, and determine turn
    board_values = board[0] + board[1] + board[2]
    
    count_x = board_values.count(X)
    count_o = board_values.count(O)
    
    return O if count_x > count_o else X



def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # If game is over, don't return actions
    if terminal(board): 
       return "Game Over"
    
    # Keep track of possible actions
    actions = set()
    
    # Find all empty cells, and add them as possible actions
    for i, row in enumerate(board): 
        for j, cell in enumerate(row): 
            if cell == EMPTY:
                actions.add((i, j)) 
                
    return actions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Verify action is possible
    if action not in actions(board):
         
        # raise ActionNotPossible Error if not possible
        raise ActionNotPossibleError(action, "Action {} not possible!")
    
    # Else, deep copy board to create new one
    new_board = deepcopy(board)
    
    # Get inputs from action and player turn
    row = action[0]
    cell = action[1]
    symbol = player(board)
    
    # Apply action based on player turn (X or O)
    new_board[row][cell] = symbol
    
    return new_board


    
def check_if_won(board, player):
    
    # Keep track of player's actions on the board
    actions = set()
    
    for i, row in enumerate(board): 
        for j, cell in enumerate(row): 
            if cell == player:
                actions.add((i, j))

    for n in range(3):
        
        # Check if any win vertically (3 moves with same col index)
        same_col = list(filter(lambda x: x[1] == n ,actions))
        
        if len(same_col) == 3: 
            return True
        
        # Check if any win horizontally (3 moves with same row index)
        same_row = list(filter(lambda x: x[0] == n ,actions))
        
        if len(same_row) == 3: 
            return True

    # Check if any win diagonally
    
    # >> Check Option 1: (0,0), (1,1), (2,2)
    if all(x in actions for x in [(0,0), (1,1), (2,2)]):
        return True

    # >> Check Option 2: (2,0), (1,1), (0,2)
    if all(x in actions for x in [(2,0), (1,1), (0,2)]):
        return True
    
    # Else, return not won
    return False



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if check_if_won(board, X):
        return X
    if check_if_won(board, O):
        return O
    # Else
    return "No Winner"



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Count all board cells with 'O' or 'X'
    board_values = list(filter(lambda y: y == X or y == O, board[0] + board[1] + board[2]))
    
    # Check if all board cells are filled
    if len(board_values) == 9: 
        return True
    
    # Check if there is a winner
    if winner(board) is X or winner(board) is O: 
        return True

    # Else game is not over
    return False



def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    utility = 0
    if winner(board) == X:
        utility = 1
    if winner(board) == O:
        utility = -1

    return utility



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    total_time = time.time()
    
    # If game is over, no more possible actions
    if terminal(board): 
       return None   

    # Keep track of alpha and beta for pruning
    alpha = -math.inf
    beta = math.inf
    
    if player(board) == X:
        best_score = -math.inf
        best_move = None

        for action in actions(board):
            start_time = time.time()
            print(f"\nMax AI is evaluating action {action}..")
            
            
            
            # Calculate utility of action
            res = min_value(result(board,action), alpha, beta)
            
            # Keep track of alpha
            alpha = max(alpha, res)
            print(f"Alpha: {alpha}")
            
            # Keep track of best move among possible actions
            best_move = action if best_score < res else best_move
            
            # Keep track of best score among possible actions
            best_score = max(best_score, res)
            print("\n>  %.3gs\n" % (time.time() - start_time))


        print("> Total %.3gs\n" % (time.time() - total_time))
        return best_move
        
    else:
        best_score = math.inf
        best_move = None
        
        for action in actions(board):
            start_time = time.time()
            
            print(f"\nMin AI is evaluating action {action}..")
            
            # Calculate utility of action
            res = max_value(result(board,action), alpha, beta)
            
            # Keep track of beta
            beta = min(beta, res)
            print(f"Beta: {beta}")

            # Keep track of best move among possible actions
            best_move = action if best_score > res else best_move
            
            # Keep track of best score among possible actions
            best_score = min(best_score, res)
            print("\n>  %.3gs\n" % float((time.time() - start_time)))
        
        print(">  Total %.3gs\n" % (time.time() - total_time))
        return best_move      
        
        

def max_value(board, alpha, beta):
    
    if terminal(board): 
        return utility(board)
    
    best_score = -math.inf
    
    for action in actions(board):   
        best_score = max(best_score, min_value(result(board, action), alpha, beta))
        
        # Beta pruning: If higher than beta, don't bother.
        if best_score >= beta:
            break 
        
    return best_score



def min_value(board, alpha, beta):
    
    if terminal(board): 
        return utility(board)
    
    best_score = math.inf
    
    for action in actions(board):
        best_score = min(best_score, max_value(result(board,action), alpha, beta))
        
        # Alpha pruning: If lower than alpha, don't bother.
        if best_score <= alpha:
            break             
    
    return best_score



# Helper Function to Play Game in Terminal since PyGame did not work initially
def play_game(board):
    
    # Intro Sequence
    print("\n 🤠 Welcome to Tic Tac Toe!", end="")
    time.sleep(1.5)
    print("\r ➡️  You are player O               ", end="")
    time.sleep(1.5)
    print("\r 🚀 Let's get the game started!", end="")
    time.sleep(1.5)
    
    # Print Initial Board
    print("\r🔸 Starting Board                  \n-------------------------\n")
    time.sleep(0.1)
    for i, row in enumerate(board):
        time.sleep(0.1)
        for j, cell in enumerate(row): 
            if cell == None:
                print(colored("-", "yellow"), end="  |  ")
            else:
                print(cell, end="  |  ")
        print("\n")

    move = ""
    
    # as long as board is not terminal, keep playing
    while not terminal(board):
        
        turn = player(board)
        
        # AI starts as X
        if turn == X:
            time.sleep(2)
            print("\n>> It's the AI's turn... player", player(board))
            time.sleep(0.5)
            move = minimax(board)
            time.sleep(0.5)
            print(">> AI chose move", move)
            

        # Human is O
        else: 
            time.sleep(2)
            print("\n>> It's the Human's turn... player", player(board), "\n")
            time.sleep(0.5)
            print("🔸 Possible Options\n-------------------------\n")
            time.sleep(0.1)
            option_board = deepcopy(board)
            
            # List options
            possible_actions = enumerate(actions(board))
            for index, action in possible_actions:
                row = action[0]
                cell = action[1]
                option_board[row][cell] = index + 1
            
            # Print options board
            for i, row in enumerate(option_board):
                time.sleep(0.1)
                for j, cell in enumerate(row): 
                    if (i, j) == move: 
                        print(colored(cell, "red"), end="  |  ")
                    elif isinstance(cell, int) :
                        print(colored(cell, "yellow"), end="  |  ")
                    else:
                        print(cell, end="  |  ")
                print("\n")
            
            # Get user input
            move = list(actions(board))[int(input("\n>> Select move from above: "))-1]
            print(">> You chose move", move)

        # Update board and print it to screen
        board = result(board, move)
        
        # Print time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        # Wait before printing it to screen
        time.sleep(1)
        print(f"\n🔸 {current_time} <> New Board\n-------------------------\n")
        time.sleep(0.1)
        # print board
        for i, row in enumerate(board):
            time.sleep(0.1)
            for j, cell in enumerate(row): 
                if (i, j) == move: 
                    print(colored(cell, "red"), end="  |  ")
                elif cell == None:
                    print("-", end="  |  ")
                else:
                    print(cell, end="  |  ")
            print("\n")
        
    time.sleep(1)
    if winner(board) == "No Winner": 
        print("\n😵 No Winner! the AI is unbeatable!\n")
        print("\n------------------------ \n        GAME OVER       \n------------------------ \n")
    elif winner(board) == O:
        print("\n🎉 Congrats, Player ! You won!", winner(board), "! \n-------------------------\n")
    else:
        print("\n🤬 You lost! the 🤖 AI kicked your butt\n ")
        print("\n------------------------ \n        GAME OVER       \n------------------------ \n")
            

# Play game in terminal
#play_game(initial_state())


    
    
    