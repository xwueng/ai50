from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")
ABoth = And(AKnight, AKnave)
AType = And(Or(AKnight, AKnave), Not(ABoth))
AStatement = ABoth

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")
BBoth = And(BKnight, BKnave)
BType = And(Or(BKnight, BKnave), Not(BBoth))

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")
CBoth = And(CKnight, CKnave)
CType = And(Or(CKnight, CKnave), Not(CBoth))


# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    AType,
    Implication(AKnight, AStatement)
)

# Puzzle 1
#should adda maybe
# A says "We are both knaves."
# B says nothing.
ASaid = And(AKnave, BKnave)
knowledge1 = And(
    AType,
    BType,
    Implication(AKnight, ASaid),
    Implication(AKnave, Not(ASaid))
)


# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
ABSame = Or(And(AKnight, BKnight), And(AKnave, BKnave))
ASaid = ABSame
ABDiff = Or(And(AKnight, BKnave), And(AKnave, BKnight))
BSaid = ABDiff

knowledge2 = And(
    Or(ABSame, ABDiff),
    Not(And(ABSame, ABDiff)),
    Implication(AKnight, ASaid),
    Implication(AKnave, ABDiff),
    Implication(BKnight, BSaid),
    # Implication(BKnave, ABSame)
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
ASaid = Or(AKnight, AKnight)
AAdmitKnave = Implication(AKnight, AKnave)


knowledge3 = And(
    # AType,
    Not(AAdmitKnave),
    BType,
    Implication(BKnight, AAdmitKnave),
    CType,
    Implication(CKnight, AKnight),
    Implication(CKnave, AKnave)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
