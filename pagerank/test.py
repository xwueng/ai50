from pagerank import *

damping_factor = DAMPING
ACCURACY = 0.001

page = '1.html'
corpus00 = {"1.html": {"2.html", "3.html"},
            "2.html": {"3.html"}, "3.html": {"2.html"}}
corpus0 = {'4.html': {'2.html'}, '3.html': {'2.html', '4.html'}, '2.html': {'3.html', '1.html'}, '1.html': {'2.html'}}
corpus1 = {'bfs.html': {'search.html'}, 'dfs.html': {'search.html',
            'bfs.html'}, 'minimax.html': {'search.html', 'games.html'},
           'search.html': {'minimax.html', 'bfs.html', 'dfs.html'},
           'tictactoe.html': {'minimax.html', 'games.html'},
           'games.html': {'minesweeper.html', 'tictactoe.html'},
           'minesweeper.html': {'games.html'}}
corpus2 = {'python.html': {'programming.html', 'ai.html'},
           'c.html': {'programming.html'},
           'logic.html': {'inference.html'},
           'programming.html': {'c.html', 'python.html'},
           'inference.html': {'ai.html'},
           'algorithms.html': {'programming.html', 'recursion.html'},
           'recursion.html': set(),
           'ai.html': {'inference.html', 'algorithms.html'}}
nooutgoing = {'5.html': set(), '4.html': {'2.html', '5.html'}, '3.html': {'2.html', '4.html'}, '2.html': {'3.html', '1.html'}, '1.html': {'2.html'}}
noincoming = {'5.html': {'4.html'}, '4.html': {'2.html'}, '3.html': {'2.html', '4.html'}, '2.html': {'3.html', '1.html'}, '1.html': {'2.html'}}
nooutgoing2 = {'5.html': {'1.html', '2.html', '3.html', '4.html'}, '4.html': {'2.html', '5.html'}, '3.html': {'2.html', '4.html'}, '2.html': {'3.html', '1.html'}, '1.html': {'2.html'}}

collections = (corpus0, corpus1, corpus2)
n = SAMPLES

def check_delta(corpus, golden_iter):
    ranks = iterate_pagerank(corpus, damping_factor)
    print("--- check deltas ---")
    for rank in ranks:
        delta = abs(golden_iter[rank] - ranks[rank])
        if delta > ACCURACY:
            print(f"Error:  {rank}: ", '%.4f' % ranks[rank], " delta: ", '%.4f' % delta, "  exceeds ", ACCURACY)
        else:
            print(f"{rank}: ", '%.4f' % ranks[rank], " delta: ", '%.4f' % delta)
#--- test sample ---
# for corpus in collections:
#     print(f"\ntest: iterate_pagerank corpus under test: {corpus}")

# --- test accuracy against course answers ---
golden_sample0= {'1.html' : 0.2223, '2.html' : 0.4303, '3.html' : 0.2145, '4.html' : 0.1329}
golden_iter0= {'1.html' : 0.2202, '2.html' : 0.4289, '3.html' : 0.2202, '4.html' : 0.1307}
corpus = corpus0
golden_iter = golden_iter0
check_delta(corpus, golden_iter)

golden_iter1 = {'bfs.html': 0.1152, 'dfs.html': 0.0809,
                'minimax.html': 0.1312, 'search.html': 0.2090,
                'tictactoe.html': 0.1180, 'games.html': 0.2277,
                'minesweeper.html': 0.1180}
corpus = corpus1
golden_iter = golden_iter1
check_delta(corpus, golden_iter)

golden_iter2 = {'python.html': 0.1243, 'c.html': 0.1243,
                'logic.html': 0.0264, 'programming.html': 0.2293,
                'inference.html': 0.1291, 'algorithms.html': 0.1067,
                'recursion.html': 0.0716, 'ai.html': 0.1884}

corpus = corpus2
golden_iter = golden_iter2
check_delta(corpus, golden_iter)



