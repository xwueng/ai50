import sys

from crossword import *
from collections import deque
from operator import itemgetter


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
        return self.backtrack(dict())
    # ...
    def letter_grid_incomplete(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            if word is not None:
                for k in range(len(word)):
                    i = variable.i + (k if direction == Variable.DOWN else 0)
                    j = variable.j + (k if direction == Variable.ACROSS else 0)
                    letters[i][j] = word[k]
        return letters

    def print_incomplete_assignment(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid_incomplete(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def find_dupe_values(self, assignment: dict):
        dupe_dict = {}
        for var, value in assignment.items():
            if value != None:
                dupe_dict[value] = set()

        for var, value in assignment.items():
            if value != None:
                dupe_dict[value].add(var)

        for value, vars in dupe_dict.items():
            if len(vars) > 1:
                return True
        return False

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        for var in self.crossword.variables:
            varlen = var.length
            domain = self.domains[var].copy()
            for word in domain:
                if len(word) != varlen:
                    self.domains[var].remove(word)

    def get_chars(self, var: Variable, index, domains=None):
        # var is a variable
        chars = set()
        # if domains is None:
        words = self.domains[var]

        for word in words:
            if len(word) - 1 >= index:
                chars.add(word[index])
        return chars

    def find_domain_conflict(self, x, y, domains=None):
        overlap = self.crossword.overlaps[(x, y)]
        missing_in_y = None
        found_conflict = False
        if overlap is not None:
            xindex, yindex = overlap
            xchars = self.get_chars(x, xindex, domains)
            ychars = self.get_chars(y, yindex, domains)
            missing_in_y = xchars - ychars
            found_conflict = len(missing_in_y) > 0
        return (found_conflict, xindex, missing_in_y)

    def find_value_conflict(self, x, xword, y, domains=None):
        overlap = self.crossword.overlaps[(x, y)]
        conflict_chars = set()
        xindex = 0
        found_conflict = False
        if overlap is not None:
            xindex, yindex = overlap
            xchar = set(xword[xindex])
            ychars = self.get_chars(y, yindex, domains)
            conflict_chars = ychars - xchar
            found_conflict = conflict_chars is not None
        return (found_conflict, xindex, conflict_chars)

    def assignment_consistent(self, x, y, assignment):
        overlap = self.crossword.overlaps[(x, y)]
        if not (x and y in assignment.keys()):
            return True
        xword = assignment[x]
        yword = assignment[y]

        if (overlap and xword and yword) is not None:
            xindex, yindex = overlap
            xchar = xword[xindex]
            ychar = yword[yindex]
            return xchar == ychar
        else:
            return True

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        revised = False
        found_domain_conflict, xindex, missing_in_y = self.find_domain_conflict(x, y)
        if found_domain_conflict:
            domain_copy = self.domains[x].copy()
            for word in domain_copy:
                if len(word) - 1 >= xindex:
                    if word[xindex] in missing_in_y:
                        self.domains[x].remove(word)
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

        if arcs is None:
            var_pairs = set()
            unique_overlaps = list()

            for var_pair, value in self.crossword.overlaps.items():
                v1, v2 = var_pair
                if value is not None and ((v1, v2) not in var_pairs) and ((v2, v1) not in var_pairs):
                    var_pairs.add(var_pair)
                    unique_overlaps.append(var_pair)
            queue = deque(unique_overlaps)
        else:
            queue = deque(arcs)

        while len(queue) >= 2:
            x, y = queue.pop()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for neighbor in (self.crossword.neighbors(x) - {y}):
                    queue.append((neighbor, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment) == 0:
            return False
        else:
            for a in assignment:
                if assignment[a] is None:
                    return False
            return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        self.enforce_node_consistency()

        # 1. all dict values are unique
        if self.find_dupe_values(assignment):
            return False

        # 3. no confict with neighboring variables
        for var, value in assignment.items():
            if value is not None:
                neighbors = self.crossword.neighbors(var)
                neighbor_arcs = list()
                # check neighbor consistency
                for neighbor in neighbors:
                    if not self.assignment_consistent(var, neighbor, assignment):
                        return False
                    neighbor_arcs.append((var, neighbor))
                if not self.ac3(neighbor_arcs):
                    return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # return self.domains[var]
        sorted_stats = list()

        neighbors = self.crossword.neighbors(var)
        unassigned_neighbors = set()
        for neighbor in neighbors:
            if neighbor in assignment.keys():
                if assignment[neighbor] is None:  # track unassigned neighbors only
                    unassigned_neighbors.add(neighbor)

        for value in self.domains[var]:
            nconflicts = 0
            nneighbor_impacted = 0
            if len(value) == var.length:
                for neighbor in unassigned_neighbors:
                    found_value_conflict, xindex, conflict_chars = \
                        self.find_value_conflict(var, value, neighbor, self.domains)
                    if found_value_conflict:
                        nconflicts += len(conflict_chars)
                        nneighbor_impacted += 1

                    sorted_stats.append((value, nconflicts, nneighbor_impacted))

        if len(sorted_stats) > 0:
            sorted_stats = sorted(sorted_stats, key=itemgetter(1))
            sorted_stats = sorted(sorted_stats, key=itemgetter(2))

            sorted_domain_values = set()
            for value, nconflicts, nneighbor_impacted in sorted_stats:
                sorted_domain_values.add(value)
            return sorted_domain_values
        else:
            return self.domains[var]

    def select_unassigned_variable(self, assignment):
        unassigned = list()
        for var, value in assignment.items():
            if value is None:
                nvalues = len(self.domains[var])
                nneighbors = len(self.crossword.neighbors(var))
                unassigned.append((var, nvalues, nneighbors))

        if len(unassigned) > 0:
            unassigned = sorted(unassigned, key=itemgetter(1))
            unassigned = sorted(unassigned, key=itemgetter(2), reverse=True)
            var, nvalues, nneighbors = unassigned[0]
            return var
        else:
            return None

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if len(assignment) == 0:
            assignment = dict()
            for var in self.domains:
                assignment[var] = None

        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            assignment[var] = value

            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is None:
                    assignment.pop(var)
                else:
                    return result
        return None

    def test(self):
        x = Variable(0, 1, 'down', 5)
        y = Variable(0, 1, 'across', 3)
        self.revise(x, y)
        word = 'THREE'
        assignmentx = {x: word}
        word = 'TEN'
        assignmenty = {y: word}
        assignment = dict()
        for var in self.domains:
            assignment[var] = None

        assignment[y] = 'TEN'

        # arc = {self.crossword.overlaps[(x, y)]: None}
        # arc = (x, y)
        # print(f"creator.ac3(arc): {self.ac3(arc)}")

        # self.domains = {Variable(0, 1, 'down', 5): {'THREE', 'SEVEN'}, Variable(4, 1, 'across', 4): {'NINE'}, Variable(1, 4, 'down', 4): {'FIVE', 'NINE'}, Variable(0, 1, 'across', 3): {'TEN', 'SIX', 'TWO'}}
        # print(f"creator.ac3(arc): {self.ac3()}")

        # print(f"creator.assignment_complete(assignmentx): {self.assignment_complete(assignmentx)}")
        # print(f"creator.consistent(assignmentx): {self.consistent(assignmentx)}")
        # print(f" var: {x}, assignment: {assignmentx}, order: {self.order_domain_values(x, assignmentx)}")
        print(f" var: {y}, assignment: {assignment}, order: {self.order_domain_values(y, assignment)}")
        # self.select_unassigned_variable()
        # assignment = {Variable(0, 1, 'down', 5): None, Variable(0, 1, 'across', 3): 'SIX', Variable(1, 4, 'down', 4): 'NINE', Variable(4, 1, 'across', 4): None}
        # self.backtrack(assignment)

        # self.print(assignmentx)
        # temp = {Variable(4, 1, 'across', 4): 'NINE', Variable(1, 4, 'down', 4): 'FIVE', Variable(0, 1, 'across', 3): 'SIX', Variable(0, 1, 'down', 5): 'SEVEN'}
        # var = Variable(0, 1, 'down', 5) #SEVEN
        # neighbor = Variable(4, 1, 'across', 4) #NINE
        #
        # consistent = self.assignment_consistent(var, neighbor, temp)

    # ...

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
