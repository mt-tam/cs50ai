import itertools
import random
import sys
from termcolor import colored, cprint


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        mines = []
        
        if self.count == len(self.cells): 
            for cell in self.cells:
                mines.append(cell)

        return mines

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        safes = []

        if self.count == 0: 
            for cell in self.cells:
                safes.append(cell)

        return safes

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        self.cells.remove(cell)            
        self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        sentences_to_update = list(filter(lambda sentence: cell in sentence.cells, self.knowledge))

        for sentence in sentences_to_update:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        sentences_to_update = list(filter(lambda sentence: cell in sentence.cells, self.knowledge))
        
        for sentence in sentences_to_update:
            sentence.mark_safe(cell)



    def check_subset(self, sentence1):
        response = False
        for sentence2 in self.knowledge:
            if sentence1.cells.issubset(sentence2.cells) and sentence1 != sentence2 and len(sentence1.cells) != 0:
                
                #print(f"    {sentence1} is a subset of {sentence2}.")

                # Update the top set
                sentence2.cells = sentence2.cells - sentence1.cells
                sentence2.count = sentence2.count - sentence1.count
                
                #print(f"    Sentence updated: {sentence2}")
                response = True
        
        return response



    def check_known(self, sentence):

        known_mines = sentence.known_mines()
        known_safes = sentence.known_safes()

        response = False

        # If there are known mines, remove them from sentence
        if len(known_mines) > 0:
            for cell in known_mines:
                self.mark_mine(cell) 
            response = True        

        # If there are known safes, remove them from sentence
        if len(known_safes) > 0:
            for cell in known_safes:
                self.mark_safe(cell)
            response = True

        # If all cells were removed, then remove sentence from knowledge
        if len(sentence.cells) == 0: 
            self.knowledge.remove(sentence)

        return response


    def add_knowledge(self, cell, count):
        
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)
        
        # 2) mark the cell as safe
        self.mark_safe(cell)

        # 3) add a new sentence to the AI's knowledge base based on the value of `cell` and `count`

        # Get all surrounding cells 
        nearby_cells = []

        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds
                if 0 <= i < self.height and 0 <= j < self.width:                    
                    nearby_cells.append((i, j))

        # Refine search and get all valid cells
        valid_cells = []

        for n in nearby_cells: 
            
            # Remove known safe moves
            if n in self.safes:
                continue

            # Remove known mines
            if n in self.mines:
                count -= 1
                continue
            
            valid_cells.append(n)
                
                
        # Ad temp sentence to knowledge if something left
        new_sentence = Sentence(valid_cells, count)
        self.knowledge.append(new_sentence)

        for sentence in self.knowledge:
            self.check_known(sentence)
            self.check_subset(sentence)

        # Latest Knowledge:
        #print("\n ------ Updated Sentences ------\n")
        #if len(self.knowledge) == 0: 
        #    print("     -none-")
        #for sentence in self.knowledge:
        #    print(f"    🔸 {sentence}")


        print("\n ------ Known Moves ------\n")
        #print(f"    Past Moves: {self.moves_made}")
        print(f"    Safe Moves: {self.safes - self.moves_made}")
        print(f"    Known Mines: {self.mines}\n\n")     
        
    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for cell in self.safes:
            if cell in self.moves_made:
                continue
            else:
                print(f"Safe Move: {cell}")
                return cell

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.moves_made:
                    continue
                if (i, j) in self.mines:
                    continue   
                print(f"Random Move: {(i, j)}")             
                return (i, j)
