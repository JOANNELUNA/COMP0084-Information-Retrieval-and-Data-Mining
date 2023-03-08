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


def get_vocabulary_size(filename, remove = False):
    vocabulary = set()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = word_tokenize(line.lower())
            for token in tokens:
                lemma = lemmatizer.lemmatize(token)
                if remove == False:
                    if lemma.isalpha():                    
                        vocabulary.add(lemma)
                elif remove == True:
                    if lemma.isalpha() and lemma not in stop_words:
                        vocabulary.add(lemma)
    return len(vocabulary)
    
print('stop words remained, the length is',(get_vocabulary_size('passage-collection.txt')))
print('stop words removed, the length is',(get_vocabulary_size('passage-collection.txt', remove = True)))



def count_terms(filename, remove = False):
    term_counts = {}
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = word_tokenize(line.lower())
            for token in tokens:
                lemma = lemmatizer.lemmatize(token)
                if remove == False:
                    if lemma.isalpha():
                        term_counts[lemma] = term_counts.get(lemma, 0) + 1
                elif remove == True:
                    if lemma.isalpha() and lemma not in stop_words:
                        term_counts[lemma] = term_counts.get(lemma, 0) + 1
    return term_counts

term_counts = count_terms('passage-collection.txt')
term_counts_remove = count_terms('passage-collection.txt', remove = True)


def plot_zipf(term_counts, term_counts_remove, remove = False):
    if remove == False:
        total_count = sum(term_counts.values())
        term_freq = [(term_counts[term] / total_count) for term in term_counts] # norm freq
    elif remove == True:
        total_count = sum(term_counts_remove.values())
        term_freq = [(term_counts_remove[term] / total_count) for term in term_counts_remove] # norm freq

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
    if remove == False:
        plt.savefig('plot_zipf.png')
    elif remove == True:
        plt.savefig('plot_zipf_remove.png')
    

def Diff(term_counts, term_counts_remove, remove = False):
    if remove == False:
        total_count = sum(term_counts.values())
        term_freq = [(term_counts[term] / total_count) for term in term_counts] # norm freq
    elif remove == True:
        total_count = sum(term_counts_remove.values())
        term_freq = [(term_counts_remove[term] / total_count) for term in term_counts_remove] # norm freq

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
    if remove == False:
        plt.savefig('Diff.png')
    elif remove == True:
        plt.savefig('Diff_remove.png')
    
plot_zipf(term_counts, term_counts_remove)
Diff(term_counts, term_counts_remove)
plot_zipf(term_counts, term_counts_remove, remove = True)
Diff(term_counts, term_counts_remove, remove = True)