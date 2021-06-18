import itertools
import random
import copy


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
        if self.count == len(self.cells):
            return self.cells
        else:
            return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        else:
            return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.count -= 1
            self.cells.remove(cell)

        if self.count > 0 and len(self.cells) == 0:
            print(f" WARNING: sentence mark_mine: count={self.count}, cell={cell}  \
                   but cells are empty: {self.cells}")

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)

        if self.count > 0 and len(self.cells) == 0:
            print(f"WARNING: sentence mark_safe: after marking/removing safe cell: sentence.count={self.count}, cell makred ={cell} \
                             but cells are empty: {self.cells}")


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

    def print(self):
        print(" AI contents:")
        print(f" AI moves_made: {self.moves_made} \n mines: {self.mines} \n safes:{self.safes}")
        for sentence in self.knowledge:
            print(f"sentence: {sentence}")

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def remove_mine_sentences(self):
        # self is an AI
        # add mines to AI.mines if sentence count ==  number of cells in a sentence
        knowledge_copy = copy.deepcopy(self)
        for sentence in knowledge_copy.knowledge:
            mine_cells = sentence.known_mines()
            if len(mine_cells) > 0:
                for mine in mine_cells:
                    self.mark_mine(mine)

    def remove_safe_sentences(self):
        # make all cells mines if sentence count ==  number of cells in a sentence
        knowledge_copy = copy.deepcopy(self)
        for sentence in knowledge_copy.knowledge:
            if len(sentence.cells) == 0:
                self.knowledge.remove(sentence)
            else:
                safe_cells = sentence.known_safes()
                if len(safe_cells) > 0:
                    for safe in safe_cells:
                        self.mark_safe(safe)
                    # remove the sentence from knowledge since
                    # all cells in this sentence are resolved to be safe
                    # self.knowledge.remove(sentence)

    def get_neighbors(self, cell):
        neighbors = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                if 0 <= i < self.height and 0 <= j < self.width:
                    neighbors.add((i, j))

        return neighbors

    def mark_neighbors(self, neighbors: set, mine_count):
        # self is an ai
        remaining_mines = mine_count
        len_neighbors = len(neighbors)
        unmarked_neighbors = set()
        mines_and_safes = self.safes.union(self.mines)

        if len_neighbors > 0:
            if mine_count == 0:
                print(f"mark_neighbors: mark all neigbors safe, "
                      f"mine_count = {mine_count}, neighbors {neighbors}")
                for neighbor in neighbors:
                    if neighbor not in self.safes:
                        self.mark_safe(neighbor)
            elif mine_count > 0 and mine_count == len_neighbors:
                print(f"mark_neighbors: mark all neigbors mine, "
                      f"mine_count = {mine_count}, neighbors {neighbors}")
                for neighbor in neighbors:
                    if neighbor not in self.mines:
                        self.mark_mine(neighbor)
                    remaining_mines = 0
            else:
                for neighbor in neighbors:
                    if neighbor not in mines_and_safes:
                        unmarked_neighbors.add(neighbor)
                    elif neighbor in self.mines:
                        remaining_mines -= 1
                print(f"mark_neighbors: , unmarked_neighbors"
                      f" mine_count: {mine_count}, original neighbors: {neighbors}, "
                      f"unmarked: {unmarked_neighbors}, remaining_mines: {remaining_mines}")
        return (unmarked_neighbors, remaining_mines)

    def add_stentence(self, neighbors: set, mine_count):
        if mine_count > 0 and len(neighbors) > 0:
            new_sentence = Sentence(neighbors, mine_count)
            if new_sentence not in self.knowledge:
                self.knowledge.append(new_sentence)
                print(f"add_stentence: add new sentence {new_sentence}")

    # def get_unmarked_neighbors(self, cell, count):
    #     x, y = cell
    #     neighbors = set()
    #     width = self.width
    #     height = self.height
    #     rstart = max(0, x - 1)
    #     rstop = min(height, x + 2)
    #     cstart = max(0, y - 1)
    #     cstop = min(width, y + 2)
    #     makred_neighbor_mines = 0
    #     for r in range(rstart, rstop):
    #         for c in range(cstart, cstop):
    #             neighbor = (r,c)
    #             if cell == neighbor: continue
    #
    #             #here we care about only the neighbors that hasn't been revealed
    #             if neighbor not in self.moves_made:
    #                 if neighbor in self.mines:
    #                     makred_neighbor_mines += 1
    #                     continue
    #                 elif neighbor in self.safes:
    #                     continue
    #                 else:
    #                     neighbors.add(neighbor)
    #     return neighbors, makred_neighbor_mines

    def find_subset(self):
        knowledge_copy = copy.deepcopy(self.knowledge)

        found_subset = False
        # for subset_candidate in self.knowledge:
        for subset_candidate in knowledge_copy:
            sub_cells = subset_candidate.cells
            knowledge_copy.remove(subset_candidate)
            found_subset = False

            # any time we have two sentences set1 = count1 and set2 = count2
            # where set1 is a subset of set2, then we can construct the
            # new sentence set2 - set1 = count2 - count1.
            if len(knowledge_copy) > 0:
                for sentence in knowledge_copy:
                    sentence_cells = sentence.cells

                    if len(sentence_cells) > 0:
                        found_subset = sub_cells.issubset(sentence_cells) or \
                            sentence_cells.issubset(sub_cells)
                        if found_subset:
                            return found_subset
        return found_subset

    def expand_knowledge_from_inferences(self):
        knowledge_copy = copy.deepcopy(self.knowledge)

        # for subset_candidate in self.knowledge:
        for subset_candidate in knowledge_copy:
            sub_cells = subset_candidate.cells
            sub_count = subset_candidate.count
            knowledge_copy.remove(subset_candidate)
            duplicates = []

            # any time we have two sentences set1 = count1 and set2 = count2
            # where set1 is a subset of set2, then we can construct the
            # new sentence set2 - set1 = count2 - count1.
            if len(knowledge_copy) > 0:
                for sentence in knowledge_copy:
                    subset_found = False
                    diff_set = set()

                    print(f"- expand_knowledge subset_candidate: {subset_candidate}, "
                          f"sentence: {sentence}")
                    sentence_cells = sentence.cells
                    sentence_count = sentence.count
                    if sentence == subset_candidate:
                        duplicates.append(sentence)
                    elif len(sentence_cells) > 0:
                        if sub_cells.issubset(sentence_cells):
                            diff_set = sentence_cells - sub_cells
                            duplicates.append(sentence)
                            subset_found = True
                        elif sentence_cells.issubset(sub_cells):
                            diff_set = sub_cells - sentence_cells
                            duplicates.append(subset_candidate)
                            subset_found = True

                        if subset_found:
                            diff_count = abs(sentence_count - sub_count)
                            self.add_stentence(diff_set, diff_count)

            # remove duplicated sentences from knowledge
            if len(duplicates) > 0:
                for sentence in duplicates:
                    if sentence in self.knowledge:
                        self.knowledge.remove(sentence)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1) mark the cell as a move that has been made
        print(f"--- add_knolwdge move made cell: {cell}, count: {count}")
        self.moves_made.add(cell)
        print(f"step 1: move made cell: {cell}, count: {count} mark the cell as a move that has been made")
        self.print()

        # 2) mark the cell as safe
        self.mark_safe(cell)
        print(f"step 2: move made cell: {cell}, count: {count} mark the cell as safe")
        self.print()

        #  3) add a new sentence to the AI's knowledge base
        #     based on the value of `cell` and `count`
        print(f"step 3: move made cell: {cell}, count: {count} add a new sentence to the AI's knowledge base")
        neighbors = self.get_neighbors(cell)
        unmarked_neighbors, remaining_mines = self.mark_neighbors(neighbors, count)
        if len(unmarked_neighbors) > 0:
            self.add_stentence(unmarked_neighbors, remaining_mines)
        self.print()



        while True:
            #  4) mark any additional cells as safe or as mines
            # if it can be concluded based on the AI's knowledge base
            if len(self.knowledge) > 0:
                print(f"step 4: move made cell: {cell}, count: {count} "
                      f"mark any additional cells as safe or as mines")
                self.remove_mine_sentences()
                self.remove_safe_sentences()
                self.print()

            #  5) add any new sentences to the AI's knowledge base
            #  if they can be inferred from existing knowledge
            if len(self.knowledge) > 0:
                print("step 5 expand_knowledge_from_inferences")
                self.expand_knowledge_from_inferences()
                self.print()

            if not self.find_subset():
                return None

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        safes_not_moved = self.safes - self.moves_made
        if len(safes_not_moved) == 0:
            return None
        else:
            safe_move = safes_not_moved.pop()
            self.moves_made.add(safe_move)
            print(f"--- make_safe_move: {safe_move}, self.safes: {self.safes}")
            return safe_move

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        max_len = self.width * self.height
        if (len(self.moves_made) == max_len) or \
                (max_len - len(self.moves_made) == len(self.mines)):
            return None
        invalid_new__moves = self.moves_made.union(self.mines)

        while True:
            rmove = (random.randrange(self.height), random.randrange(self.width))
            if (rmove not in invalid_new__moves):
                print(f"--- make_random_move: {rmove}")
                return rmove