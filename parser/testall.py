import os
import sys
from parser import *

def main():

    data_dir = 'sentences'

    with os.scandir(data_dir) as files:
        for f in files:
            path = os.path.join(data_dir, f.name)
            print(f"\n--- file: {path}")
            with open(path) as data_file:
                s = data_file.read()
                # Convert input into list of words
                s = preprocess(s)

                # Attempt to parse sentence
                try:
                    trees = list(parser.parse(s))
                except ValueError as e:
                    print(e)
                if not trees:
                    print("Could not parse sentence.")

                # Print each tree with noun phrase chunks
                for tree in trees:
                    tree.pretty_print()

                    print("Noun Phrase Chunks")
                    for np in np_chunk(tree):
                        print(" ".join(np.flatten()))


if __name__ == "__main__":
    main()

