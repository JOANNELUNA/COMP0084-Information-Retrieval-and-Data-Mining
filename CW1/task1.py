import re
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    # Get the current working directory
    cwd = os.getcwd()
    # Construct the file path relative to the current working directory
    filepath = os.path.join(cwd, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = word_tokenize(line.lower())
            for token in tokens:
                lemma = lemmatizer.lemmatize(token)
                if lemma.isalpha():
                    term_counts[lemma] = term_counts.get(lemma, 0) + 1
    return term_counts

term_counts = count_terms('passage-collection.txt')



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


def plot_zipf(term_counts, type):

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
    if type == "term_counts":
        plt.savefig('plot_zipf.png')
    elif type == "term_counts_remove":
        plt.savefig('plot_zipf_remove.png')
    

def Diff(term_counts, type):

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
    if type == "term_counts":
        plt.savefig('Diff.png')
    elif type == "term_counts_remove":
        plt.savefig('Diff_remove.png')
    
plot_zipf(term_counts,'term_counts')
Diff(term_counts,'term_counts')
plot_zipf(term_counts_remove,'term_counts_remove')
Diff(term_counts_remove,'term_counts_remove')