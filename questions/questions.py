import nltk
import sys
import os
from nltk.corpus import stopwords
import string
import math
import re

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            files[filename] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    invalid_words = set(stopwords.words('english')).union(set(string.punctuation))
    tokens = [token.lower() for token in nltk.word_tokenize(document)]
    words = [token for token in tokens
             if token not in invalid_words]

    # words.sort()
    return words

def tokenize2(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document2 = re.sub(r'([a-z, 0-9])\.([A-Z])', r'\1. \2', document) #words 3712
    invalid_words = set(stopwords.words('english')).union(set(string.punctuation))
    tokens = [token.lower() for token in nltk.word_tokenize(document2)]
    words = [token for token in tokens
             # if token not in invalid_words]
             if token not in invalid_words and re.search('^\w+', token) is not None] #words: 2126

    words.sort()

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    docs_word_freqs = {}
    for filename, doc in documents.items():
        doc_word_freqs = nltk.FreqDist(nltk.ngrams(doc, 1))
        for word, freq in doc_word_freqs.items():
            key = list(word)[0]
            if freq > 0:
                try:
                    docs_word_freqs[key] = docs_word_freqs[key] + 1
                except:
                    docs_word_freqs[key] = 1

    len_docs = len(documents)
    for word in docs_word_freqs:
        idfs[word] = math.log(len_docs / docs_word_freqs[word])
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    docs_tf_idfs = dict()
    for doc, words in files.items():
        doc_tf_idf = 0
        doc_word_freqs = {}
        temp_word_freqs = nltk.FreqDist(nltk.ngrams(words, 1))
        for word, freq in temp_word_freqs.items():
            key = list(word)[0]
            doc_word_freqs[key] = freq

        stop_words = set(stopwords.words('english'))
        for word in query:
            if word not in stop_words:
                try:
                    doc_tf_idf += doc_word_freqs[word] * idfs[word]
                except:
                    no_tf_idf = True

        docs_tf_idfs[doc] = doc_tf_idf

    # sort docs_tf_idfs by tf-idf values
    tf_idfs = sorted(docs_tf_idfs.items(), key=lambda x: x[1], reverse=True)
    sorted_files = [item[0] for item in tf_idfs]
    return sorted_files[: n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentences_idfs = dict()
    for sent, sent_words in sentences.items():
        sent_idf = 0
        sent_word_freqs = {}
        temp_word_freqs = nltk.FreqDist(nltk.ngrams(sent_words, 1))
        for word, freq in temp_word_freqs.items():
            key = list(word)[0]
            sent_word_freqs[key] = freq

        # get current sentence's overlap with the query
        stopwords_words = set(stopwords.words('english'))
        puncs = set(string.punctuation)
        words_intersection = set(sent_words).intersection(query) \
            - stopwords_words - puncs

        try:
            query_term_density = len(words_intersection) / len(sent)
        except:
            query_term_density = 0

        # add current word's idf to current sentence's idf
        for word in words_intersection:
            try:
                sent_idf += idfs[word]
            except:
                no_idf = True

        # current sentence's value is a tuple of its idf and query term density
        sentences_idfs[sent] = (sent_idf, query_term_density)
    idfs = sorted(sentences_idfs.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
    sorted_sentences = [item[0] for item in idfs]
    return sorted_sentences[:n]


if __name__ == "__main__":
    main()
