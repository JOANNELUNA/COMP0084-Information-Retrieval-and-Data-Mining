import os
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from task1 import count_terms

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

term_counts = count_terms('passage-collection.txt')



# function to build inverted index
def build_inverted_index(filename, term_counts, remove_stopwords=False):
    inverted_index = {}
    doc_info = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            pid, query, passage = parts[0], parts[2], parts[3]
            terms = [term.lower() for term in re.findall(r'\w+', passage)]
            if remove_stopwords:
                stop_words = set(stopwords.words('english'))
                terms = [term for term in terms if term not in stop_words]
            doc_len = len(terms)
            doc_info[pid] = (terms, doc_len)
            for term in set(terms):
                if term in term_counts:
                    if term not in inverted_index:
                        inverted_index[term] = []
                    inverted_index[term].append((pid, terms.count(term)))
    return inverted_index, doc_info

build_inverted_index('candidate-passages-top1000.tsv',term_counts)