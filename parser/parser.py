import nltk
import sys
from collections import defaultdict

TERMINALS = """
# A -> 'small' | 'white'
# N -> 'cats' | 'trees'
# V -> 'climb' | 'run'
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
# S -> NP V
# NP -> N | A NP

S -> NP VP | NP VP PP | NP VP NP | NP VP NP PP | S Conj S

AP -> Adj | Adj Adv | Adv
NP -> Adj N | AP N | Det N | | Det Adj N | Det AP N  | N | AP NP | D NP 
PP -> P | P NP | P Det N | Det PP
VP -> V | V Adv | V NP | V NP PP | V Det NP | V Det Adj NP | VP Conj VP | VP PP | Adv VP | VP Adv

"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = []
    words.extend([
        word.lower() for word in nltk.word_tokenize(sentence)
        if any(c.isalpha() for c in word)
    ])
    return words


def get_children_labels(tree, children_labels:dict):
    try:
        tree.label()
    except AttributeError:
        no_label = True
    else:
        # Now we know that tree.node is defined
        children_labels[tree.label()] += 1
        for child in tree:
            get_children_labels(child, children_labels)


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    NPs = []

    for subtree in tree.subtrees():
        if subtree.label() == "NP" and len(subtree) > 0:
            # initialize children labels counts to 0
            children_labels = {}
            children_labels = defaultdict(lambda: 0, children_labels)
            get_children_labels(subtree, children_labels)
            if children_labels['NP'] == 1:
                NPs.append(subtree)
    return NPs


if __name__ == "__main__":
    main()
