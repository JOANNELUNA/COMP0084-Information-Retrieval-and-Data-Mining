import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np


nltk.download('punkt')
nltk.download('wordnet')

def get_vocabulary_size(filename):
    vocabulary = set()
    lemmatizer = WordNetLemmatizer()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = word_tokenize(line.lower())
            for token in tokens:
                lemma = lemmatizer.lemmatize(token)
                if lemma.isalpha():
                    vocabulary.add(lemma)
    return len(vocabulary)

def count_terms(filename):
    term_counts = {}
    lemmatizer = WordNetLemmatizer()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = word_tokenize(line.lower())
            for token in tokens:
                lemma = lemmatizer.lemmatize(token)
                if lemma.isalpha():
                    term_counts[lemma] = term_counts.get(lemma, 0) + 1
    return term_counts

term_counts = count_terms('passage-collection.txt')

def plot_zipf(term_counts):

    total_count = sum(term_counts.values())
    term_freq = [(term_counts[term] / total_count) for term in term_counts] # norm freq
    assert np.isclose(sum(term_freq),1), "frequency is not normalised"
    sorted_term_freq = sorted(term_freq, reverse=True)
    

    rank = range(1, len(sorted_term_freq) + 1)
    zipf_dist = [1/(i) for i in rank]

    plt.plot(rank, sorted_term_freq,label = 'Empirical')
    plt.plot(rank, zipf_dist, label = "Zipfian")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plot_zipf.png')
    
plot_zipf(term_counts)

def Diff(term_counts):

    total_count = sum(term_counts.values())
    term_freq = [(term_counts[term] / total_count) for term in term_counts] # norm freq
    assert np.isclose(sum(term_freq),1), "frequency is not normalised"
    sorted_term_freq = sorted(term_freq, reverse=True)
    

    rank = range(1, len(sorted_term_freq) + 1)
    zipf_dist = [1/(i) for i in rank]

    log_diff = [abs(x - y) for x, y in zip(sorted_term_freq, zipf_dist)]
    
    # ratio = [sorted_term_freq[i] / zipf_dist[i] for i in range(len(rank))]
    squared_errors = [(observed - expected) ** 2 for observed, expected in zip(sorted_term_freq, zipf_dist)]
    gof_measure = sum(squared_errors)
    print('gof_measure',gof_measure)

    plt.plot(rank, log_diff)
    # plt.plot(rank, squared_errors)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Diff')
    pplt.savefig('Diff.png')
    

Diff(term_counts)

nltk.download('stopwords')

def count_terms_remove(filename):
    term_counts = {}
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = word_tokenize(line.lower())
            for token in tokens:
                lemma = lemmatizer.lemmatize(token)
                if lemma.isalpha() and lemma not in stop_words:
                    term_counts[lemma] = term_counts.get(lemma, 0) + 1
    return term_counts
    
term_counts_remove = count_terms_remove('passage-collection.txt')

plot_zipf(term_counts_remove)
Diff(term_counts_remove)