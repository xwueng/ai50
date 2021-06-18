from heredity import *

ACCURACY = 0.001

# f0
people = {'Harry': {'name': 'Harry', 'mother': 'Lily',
        'father': 'James', 'trait': None},
        'James': {'name': 'James', 'mother': None, 'father': None, 'trait': True},
        'Lily': {'name': 'Lily', 'mother': None, 'father': None, 'trait': False}}
one_gene = {"Harry"}
two_genes = {"James"}
have_trait = {'James'}
golden_probs = {'Harry': {'gene': {1: 0.9802}, 'trait': {False: 0.431288}},
                'James': {'gene': {2: 0.01}, 'trait': {True: 0.0065}},
                'Lily': {'gene': {0: 0.96}, 'trait': {False: 0.9504}}}
golden_jp = 0.0026643247488
p = joint_probability(people, one_gene, two_genes, have_trait)
print(p)
from_parents_probs = {'Lily': {0: 0.01}, 'James': {2: 0.99}}
children_probs = {'Harry': {1: 0}}
child = 'Harry'
# print(lookup_from_parent_probs(people, child, from_parents_probs))
# mp, fp = lookup_from_parent_probs(people, child, from_parents_probs)
# print(child_g0_prob(mp, fp))
# print(child_g1_prob(mp, fp))
# print(child_g2_prob(mp, fp))


# joint_probability(people, one_gene, two_genes, have_trait)

# --- test 2 -----
# one_gene = set()
# two_genes = set()
# have_trait = {'James'}
# p = joint_probability(people, one_gene, two_genes, have_trait)
# print(p)
#--- family1-----
# molly, authors are parents for: charlie, fred, ginny, ron
people = {'Arthur': {'name': 'Arthur', 'mother': None, 'father': None, 'trait': False},
          'Charlie': {'name': 'Charlie', 'mother': 'Molly', 'father': 'Arthur', 'trait': False},
          'Fred': {'name': 'Fred', 'mother': 'Molly', 'father': 'Arthur', 'trait': True},
          'Ginny': {'name': 'Ginny', 'mother': 'Molly', 'father': 'Arthur', 'trait': None},
          'Molly': {'name': 'Molly', 'mother': None, 'father': None, 'trait': False},
          'Ron': {'name': 'Ron', 'mother': 'Molly', 'father': 'Arthur', 'trait': None}}
one_gene = set()
two_genes = set()
have_trait = {'Fred'}
#--- fam1 ---
people = {'Arthur': {'name': 'Arthur', 'mother': None, 'father': None, 'trait': False}, 'Charlie': {'name': 'Charlie', 'mother': 'Molly', 'father': 'Arthur', 'trait': False}, 'Fred': {'name': 'Fred', 'mother': 'Molly', 'father': 'Arthur', 'trait': True}, 'Ginny': {'name': 'Ginny', 'mother': 'Molly', 'father': 'Arthur', 'trait': None}, 'Molly': {'name': 'Molly', 'mother': None, 'father': None, 'trait': False}, 'Ron': {'name': 'Ron', 'mother': 'Molly', 'father': 'Arthur', 'trait': None}}
golden_probabilities = {'Arthur': {'gene': {2: 0.03289878209919839, 1: 0.10349924215112996, 0: 0.8636019757496717}, 'trait': {True: 0.0, False: 1.0}}, 'Charlie': {'gene': {2: 0.0017828371319003136, 1: 0.1330865896305933, 0: 0.8651305732375064}, 'trait': {True: 0.0, False: 1.0}}, 'Fred': {'gene': {2: 0.006493390908710545, 1: 0.6486095959364807, 0: 0.3448970131548088}, 'trait': {True: 1.0, False: 0.0}}, 'Ginny': {'gene': {2: 0.00269056727628656, 1: 0.1805297356699634, 0: 0.8167796970537501}, 'trait': {True: 0.11101331767530336, False: 0.8889866823246967}}, 'Molly': {'gene': {2: 0.03289878209919838, 1: 0.10349924215112996, 0: 0.8636019757496717}, 'trait': {True: 0.0, False: 1.0}}, 'Ron': {'gene': {2: 0.0026905672762865614, 1: 0.18052973566996325, 0: 0.8167796970537502}, 'trait': {True: 0.11101331767530334, False: 0.8889866823246967}}}
compute_generations(people)
# ----- fam2 ---
people = {'Arthur': {'name': 'Arthur', 'mother': None, 'father': None, 'trait': False}, 'Hermione': {'name': 'Hermione', 'mother': None, 'father': None, 'trait': False}, 'Molly': {'name': 'Molly', 'mother': None, 'father': None, 'trait': None}, 'Ron': {'name': 'Ron', 'mother': 'Molly', 'father': 'Arthur', 'trait': False}, 'Rose': {'name': 'Rose', 'mother': 'Ron', 'father': 'Hermione', 'trait': True}}
generations = {1: {'Molly', 'Hermione', 'Arthur'}, 2: {'Ron'}, 3: {'Rose'}}
compute_generations(people)
one_gene = {'Arthur', 'Hermione', 'Molly', 'Rose', 'Ron'}
two_genes = set(),
have_trait = {'Rose', 'Molly'}
# p = joint_probability(people, one_gene, two_genes, have_trait)
# print(p)
# ----------
# {'Rose', 'Hermione', 'Molly', 'Arthur', 'Ron'}
# family1: parents: auther, molly: child: Ron. Molly and Arthur are Rose's grandparents
# family2: parents: Ron, Herminoe, child: Rose
# people = {'Arthur': {'name': 'Arthur', 'mother': None, 'father': None, 'trait': False},
#           'Hermione': {'name': 'Hermione', 'mother': None, 'father': None, 'trait': False},
#           'Molly': {'name': 'Molly', 'mother': None, 'father': None, 'trait': None},
#           'Ron': {'name': 'Ron', 'mother': 'Molly', 'father': 'Arthur', 'trait': False},
#           'Rose': {'name': 'Rose', 'mother': 'Ron', 'father': 'Hermione', 'trait': True}}
# one_gene = set()
# two_genes = set()
# have_trait = {'Rose'}
# ---------
# ----- test normalization -----
probabilities = {'Harry': {'gene': {2: 0, 1: 0, 0: 0.008852852828159997}, 'trait': {True: 0, False: 0.008852852828159997}},
                 'James': {'gene': {2: 0, 1: 0, 0: 0.008852852828159997}, 'trait': {True: 0.008852852828159997, False: 0}},
                 'Lily': {'gene': {2: 0, 1: 0, 0: 0.008852852828159997}, 'trait': {True: 0, False: 0.008852852828159997}}}
# print(f"-- test normalize function: before {probabilities}")
# normalize(probabilities)
# print(f"-- test normalize function: after {probabilities}")

# ---- end to end ---
# familiy1 result:
# Arthur:
#   Gene:
#     2: 0.0329
#     1: 0.1035
#     0: 0.8636
#   Trait:
#     True: 0.0000
#     False: 1.0000
# Charlie:
#   Gene:
#     2: 0.0018
#     1: 0.1331
#     0: 0.8651
#   Trait:
#     True: 0.0000
#     False: 1.0000
# Fred:
#   Gene:
#     2: 0.0065
#     1: 0.6486
#     0: 0.3449
#   Trait:
#     True: 1.0000
#     False: 0.0000
# Ginny:
#   Gene:
#     2: 0.0027
#     1: 0.1805
#     0: 0.8168
#   Trait:
#     True: 0.1110
#     False: 0.8890
# Molly:
#   Gene:
#     2: 0.0329
#     1: 0.1035
#     0: 0.8636
#   Trait:
#     True: 0.0000
#     False: 1.0000
# Ron:
#   Gene:
#     2: 0.0027
#     1: 0.1805
#     0: 0.8168
#   Trait:
#     True: 0.1110
#     False: 0.8890
