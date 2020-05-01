import sys
import time
import math
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        start_time = time.time()
        result = self.backtrack(dict())
        print("\n>> Solved in  %.3gs\n" % (time.time() - start_time))
        return result

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # Loop through variables
        for v in self.domains: 
            domain = self.domains[v].copy()
            
            # Loop through values in a variable's domain
            for x in domain: 

                # If value length does not match variable length, remove it
                if len(x) != v.length: 
                    self.domains[v].remove(x)
            

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        
        # Get overlap square between variables x and y
        overlaps = self.crossword.overlaps[x, y]

        # If any overlap
        if overlaps:

            # Get all words for x and y
            domain_x = self.domains[x].copy() # copy dict to be able to loop and modify at the same time
            domain_y = self.domains[y] # not modified

            # Loop through words in x's domain
            for a in domain_x: 
                
                # Track conflicts
                conflicts = 0

                # Loop through words in y's domain
                for b in domain_y: 
                    
                    # If letters on overlap do not match 
                    if a[overlaps[0]] != b[overlaps[1]]:
                        
                        # Then raise conflict
                        conflicts += 1

                # If all words in y raise conflict
                if conflicts == len(domain_y): 
                    
                    # Then remove word in x
                    self.domains[x].remove(a)

                    # Set revised to True
                    revised = True

        return revised




    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # Build queue if arcs is empty
        if arcs is None: 
            arcs = list()

            for x in self.crossword.overlaps: 
                if self.crossword.overlaps[x]: 
                    arcs.append(x)
        
        # While queue not empty
        while arcs: 
        
            # Get value from queue
            (x, y) = arcs.pop()

            # If domain of x was revised
            if self.revise(x, y): 
                
                # If no more values in x, then return false (no solution)
                if len(self.domains[x]) == 0: 
                    return False

                # Else, add neighbors to be revised again 
                for z in (self.crossword.neighbors(x) - {y}): 
                    arcs.append((z, x))

        # Check to ensure no domain is empty
        for x in self.domains: 
            if not self.domains[x]: 
                return False

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """       
        # If no. of variables and no. of assignments are not equal,then return False
        if len(assignment) != len(self.domains): 
            return False
            
        # Else print assignment, and return True
        print("\nASSIGNMENT COMPLETE\n----------------------")
        for x in assignment: 
            print(f"Var {x} -> {assignment[x]}")

        return True



    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        existing_values = []

        # For each assignment, check if it violates rules
        for a in assignment: 
            
            # assigned word
            value = assignment[a]
                
            # Check if any word violates the length of its box
            if len(value) != a.length: 
                return False

            # Check if words are distinct (don't appear twice)
            if value in existing_values:
                print(f"Word {value} has been used more than once!")
                return False
            existing_values.append(value)

            # Check if words don't overlap correctly
            neighbors = self.crossword.neighbors(a)
            
            # For all assigned words, if they're a neighbor then check if they overlap
            for b in assignment: 
                if b in neighbors:
                    overlaps = self.crossword.overlaps[a, b]
                    if overlaps: 
                        # If letters on overlap do not match, then raise conflict 
                        if value[overlaps[0]] != assignment[b][overlaps[1]]:
                            print(f"Overlap found between {assignment[a]} and {assignment[b]}.")
                            return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        # Track values and # of neighbour values they rule out
        domain_values = []

        # Get var's neighbours
        neighbors = self.crossword.neighbors(var)

        # For each of var's domain values
        for x in self.domains[var]:
            
            count_ruled_out = 0

            # For each neighbor
            for n in neighbors: 
                
                # For each domain value of neighbor
                for v in self.domains[n]:
                    
                    # If words are the same , then increase count
                    if x == v: 
                        count_ruled_out += 1

                    # If words overlap incorrectly, then increase count
                    overlaps = self.crossword.overlaps[var, n]
                    if overlaps: 
                        if x[overlaps[0]] != v[overlaps[1]]:
                            count_ruled_out += 1

            # Add results
            domain_values.append({"value" : x, "count": count_ruled_out})

        
        # Extract list of values in order of lowest count to highest
        ordered_values = []
        for x in sorted(domain_values, key=lambda k: k['count']): 
            ordered_values.append(x["value"])

        return ordered_values

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Return one unassigned variable
        unassigned_variables = []

        # Keep track of min values
        min_length = math.inf
        max_degree = -math.inf

        for x in self.domains: 
            if x not in assignment: 
                
                # Number of values in domain
                length = len(self.domains[x])
                min_length = min(min_length, length)

                # Number of relationship to other nodes
                relationships = len(self.crossword.neighbors(x))
                max_degree = max(max_degree, relationships)

                # Add to results
                unassigned_variables.append({"variable": x, "length": length, "degree": relationships })

        # Get the variables with lowest values in domain
        min_domain_vars = list(filter(lambda x: x["length"] == min_length, unassigned_variables))
        return_var = ''

        # If no tie, return variable
        if len(min_domain_vars) == 1: 
            return_var = min_domain_vars[0]["variable"]
        
        # Else get the variable with max degree
        else:
            max_degree_vars = list(filter(lambda x: x["degree"] == max_degree, min_domain_vars))
            return_var = max_degree_vars[0]["variable"] # either way always return the first value (or unique value if only one)

        return return_var

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        
        """
        

        # If assignment complete, return assignment
        if self.assignment_complete(assignment): 
            return assignment

        # Pick a variable that has not been assigned
        var = self.select_unassigned_variable(assignment)
        
        # Try a value out of the domain
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value

            print("\n")
            self.print(new_assignment)

            # If new assignment consistent
            if self.consistent(new_assignment): 
                result = self.backtrack(new_assignment)
                
                # Maintain arc consistency
                arcs = []
                for x in assignment: 
                    for n in self.crossword.neighbors(x): 
                        arcs.append((x, n))
                
                self.ac3(arcs)
                
                # If not a failure, return result
                if result is not None: 
                    return result
        return None
        
 
def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()
    
    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
