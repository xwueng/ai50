import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def from_parent_g0_prob():
    mutation = PROBS['mutation']
    pg = 0.5 * mutation + 0.5 * mutation
    # pg = 0.001
    return pg


def from_parent_g1_prob():
    mutation = PROBS['mutation']
    pg = 0.5 * (1 - mutation) + 0.5 * mutation
    # pg = 0.5
    return pg


def from_parent_g2_prob():
    # pg =  0.5 * (1 - mutation rate) + 0.5 * (1 - mutation rate) same as
    #pg = 0.99
    pg = 1 - PROBS['mutation']
    return pg


def child_g0_prob(parent1_prob, parent2_prob):
    cg = (1 - parent1_prob) * (1 - parent2_prob)
    return cg


def child_g1_prob(parent1_prob, parent2_prob):
    cg = (1 - parent2_prob) * parent1_prob + (1 - parent1_prob) * parent2_prob
    return cg


def child_g2_prob(parent1_prob, parent2_prob):
    cg = parent1_prob * parent2_prob
    return cg


def trait_prob(gene_num, person, have_trait, gene_trait_probs):
    if person in have_trait:
        have_trait_prob = gene_trait_probs[person]['gene'][gene_num] * PROBS['trait'][gene_num][True]
        gene_trait_probs[person]['trait'] = {True: have_trait_prob}
    else:
        no_trait_prob = gene_trait_probs[person]['gene'][gene_num] * PROBS['trait'][gene_num][False]
        gene_trait_probs[person]['trait'] = {False: no_trait_prob}

    return gene_trait_probs


def is_parent(people, person):
    families = list(people.values())
    found = False
    for family in families:
        found = family['mother'] == person or family['father'] == person
        if found:
            break
    return found


def compute_generations(people):
    generations = {1: set(), 2: set(), 3: set()}
    parents = set()
    children = set()
    for person in people:
        if people[person]['mother'] == None and people[person]['father'] == None:
            generations[1].add(person)
        else:
            parents.add(people[person]['mother'])
            parents.add(people[person]['father'])
            children.add(person)
    generations[2] = parents - generations[1]
    generations[3] = children - generations[2]
    # print(f"--- compute_generations: people: {people}")
    # print(f"--- compute_generations: generations: {generations}")

    return generations


def compute_parent_prob(parent, generations, gene_num, have_trait, gene_trait_probs, from_parents_probs, func_name):
    # compute the prob of parent with gene_num bad genes passing bad genes to child
    from_parent_prob = func_name
    from_parents_probs[parent] = {gene_num: from_parent_prob}
    # calculate partent's own prob of having gene_num GJB2 gene(s)
    if parent in generations[1]:
        gene_trait_probs[parent]['gene'][gene_num] = PROBS['gene'][gene_num]
        trait_prob(gene_num, parent, have_trait, gene_trait_probs)

    return (gene_trait_probs, from_parents_probs)


def lookup_from_parent_probs(people, child, from_parents_probs):
    mother = people[child]['mother']
    father = people[child]['father']
    mother_prob = list(from_parents_probs[mother].values())[0]
    father_prob = list(from_parents_probs[father].values())[0]
    return (mother_prob, father_prob)


def compute_child_prob(people, person, zero_gene, one_gene, two_genes, have_trait, from_parents_probs, gene_trait_probs):
    mother_prob, father_prob = lookup_from_parent_probs(people, person, from_parents_probs)
    if person in zero_gene:
        gn = 0
        func = child_g0_prob(mother_prob, father_prob)
    elif person in one_gene:
        gn = 1
        func = child_g1_prob(mother_prob, father_prob)
    elif person in two_genes:
        gn = 2
        func = child_g2_prob(mother_prob, father_prob)
    gene_trait_probs[person]['gene'][gn] = func
    trait_prob(gn, person, have_trait, gene_trait_probs)
    return gene_trait_probs


def joint_probability(people, one_gene, two_genes, have_trait):
    from_parents_probs = {}
    gene_trait_probs = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }
    children_probs = {}
    zero_gene = set(people) - one_gene.union(two_genes)
    generations = compute_generations(people)
    joint_prob = 1
    
    for person in generations[1].union(generations[2]):
        if person in zero_gene:
            gn = 0
            func = from_parent_g0_prob()
        elif person in one_gene:
            gn = 1
            func = from_parent_g1_prob()
        elif person in two_genes:
            gn = 2
            func = from_parent_g2_prob()
        compute_parent_prob(person, generations, gn, have_trait, gene_trait_probs,
                            from_parents_probs, func)

    if generations[2] == None:
        grange = range(3, 4)
    else:
        grange = range(2, 4)

    for i in grange:
        for person in generations[i]:
            compute_child_prob(people, person, zero_gene, one_gene,
                               two_genes, have_trait, from_parents_probs,
                               gene_trait_probs)

    # calculate joint probability
    for person in gene_trait_probs:
        if person in have_trait:
            joint_prob = joint_prob * gene_trait_probs[person]['trait'][True]
        else:
            joint_prob = joint_prob * gene_trait_probs[person]['trait'][False]
    # print(f"--- joint_probability: one_gene: {one_gene}, two_genes: {two_genes}, have_trait: {have_trait}"
    #       f"\n joint_prob: {joint_prob:.4f}")
    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    pplnames = set(probabilities.keys())
    zero_gene = pplnames - one_gene.union(two_genes)
    no_trait = pplnames - have_trait
    for person in one_gene:
        probabilities[person]['gene'][1] += p
    for person in two_genes:
        probabilities[person]['gene'][2] += p
    for person in have_trait:
        probabilities[person]['trait'][True] += p
    for person in zero_gene:
        probabilities[person]['gene'][0] += p
    for person in no_trait:
        probabilities[person]['trait'][False] += p
    # print(probabilities)


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        gene_values = probabilities[person]['gene'].values()
        gene_sum = sum(gene_values)
        if gene_sum > 0:
            for gene in probabilities[person]['gene']:
                probabilities[person]['gene'][gene] /= gene_sum
        gene_values = probabilities[person]['gene'].values()
        gene_sum = sum(gene_values)
        if gene_sum > 0:
            # if abs(1 - gene_sum) >= 0.001:
            #     print(f"---normalize probabilities: gene_sum:{gene_sum} \n{probabilities}")
            assert abs(1 - gene_sum) < 0.001

        trait_values = probabilities[person]['trait'].values()
        trait_sum = sum(trait_values)
        if trait_sum > 0:
            for trait in probabilities[person]['trait']:
                probabilities[person]['trait'][trait] /= trait_sum
        trait_values = probabilities[person]['trait'].values()
        trait_sum = sum(trait_values)
        if trait_sum > 0:
            # if abs(1 - trait_sum) >= 0.001:
            #     print(f"---normalize probabilities  trait_sum: {trait_sum}\n{probabilities}")
            assert abs(1 - trait_sum) < 0.001

        # print(f"---normalize: gene_sum: {gene_sum:.4f}, trait_sum: {trait_sum:.4f}")


if __name__ == "__main__":
    main()
