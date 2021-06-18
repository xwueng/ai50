import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    ACCURACY = 0.001
    len_corpus = len(corpus)
    # if corpus[page] is None:
    #     len_plinks = 0
    # else:
    len_plinks = len(corpus[page])
    probs = 0
    cprob = (1 - damping_factor) / len_corpus
    page_ranks = {}
    if len_plinks > 0:
        pprob = damping_factor / len_plinks
    else:
        pprob = 1 / len_corpus

    # add corpous probability to all pages.
    for thispage in corpus:
        if len_plinks == 0:
            page_ranks[thispage] = pprob
        else:
            if (thispage == page) or \
                    (thispage not in corpus[page]):
                page_ranks[thispage] = cprob
            else:
                page_ranks[thispage] = cprob + pprob
        probs += page_ranks[thispage]

    # print(f"transition_model probs: {probs}")
    if abs(1 - probs) >= ACCURACY:
        # for page in page_ranks:
        #     print(f"transition_model: crobs: {page} {page_ranks[page]}")
        raise ValueError(f"transition_model probs {probs} error {abs(1 - probs)} exceeding ACCURACY")

    return page_ranks


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ACCURACY = 0.001
    samples = list()
    page_ranks = {}
    for i in range(n):
        if i == 0:
            sample = random.choice(list(corpus))
        else:
            tm = transition_model(corpus, samples[i - 1], damping_factor)
            probs = list(tm.values())
            sample = random.choices(list(corpus.keys()), weights=probs, k=1)[0]

        samples.append(sample)

    # calculate page distributions in samples
    probs = 0
    if samples is not None:
        for page in corpus:
            page_ranks[page] = samples.count(page) / n
            probs += page_ranks[page]

    assert abs(1 - probs) < ACCURACY

    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ACCURACY = 0.001
    len_corpus = len(corpus)
    cprob = (1 - damping_factor) / len_corpus

    corpus2 = corpus.copy()
    for page in corpus2:
        if len(corpus2[page]) == 0:
            for key in corpus2.keys():
                corpus2[page].add(key)

    # temp = corpus2.copy()
    incoming_links = {}
    for page in corpus2:
        incoming_links[page] = set()
        for temp_page in corpus2:
            if page in corpus2[temp_page]:
                incoming_links[page].add(temp_page)
    # print(f"page_links: {incoming_links}")

    prev_ranks = {}
    for page in corpus2:
        prev_ranks[page] = 1 / len_corpus

    temp = 0

    while True:
        cur_ranks = {}
        probs = 0
        accurate = True

        for page in corpus2:
            link_weights = 0
            if len(incoming_links[page]) > 0:
                for from_page in incoming_links[page]:
                    num_out_links = len(corpus2[from_page])
                    if num_out_links > 0:
                        link_weights += prev_ranks[from_page] / num_out_links
                    else:
                        link_weights += prev_ranks[from_page] / len_corpus
            else:
                link_weights = prev_ranks[from_page] / len_corpus

            cur_ranks[page] = cprob + (damping_factor * link_weights)

            probs += cur_ranks[page]
            accurate = accurate and (abs(cur_ranks[page] - prev_ranks[page]) < ACCURACY)

        accurate = accurate and (abs(1 - probs) < ACCURACY)
        # print("\ninterative_ranks probs:" + '%.4f'%probs)
        # print(f"interative_ranks cur_ranks: {cur_ranks}")
        temp += 1

        if accurate:
            # assert abs(1 - probs) <= ACCURACY
            # print(f"interative_ranks probs: {probs}, temp: {temp} \n{cur_ranks}")
            return cur_ranks
        else:
            prev_ranks = copy.deepcopy(cur_ranks)


if __name__ == "__main__":
    main()
