from task1 import count_terms, count_terms_remove
from task2 import build_inverted_index
import csv
import math
from collections import defaultdict, Counter
from operator import itemgetter


def compute_doc_probs_laplace(query_terms, doc_terms, V):
    """
    Computes the log-probability of each document given the query terms
    using the query likelihood language model with Laplace smoothing.
    
    Parameters:
        query_terms (list): a list of query terms
        doc_terms (list): a list of terms from a document
        V (int): the size of the vocabulary
    
    Returns:
        float: the sum of the log-probabilities of each query term given the document
    """
    log_prob = 0
    doc_freqs = Counter(doc_terms)
    doc_len = len(doc_terms)
    for term in query_terms:
        tf = doc_freqs.get(term, 0)
        p = (tf + 1) / (doc_len + V)
        log_prob += math.log(p)
    return log_prob
    

def compute_doc_probs_lidstone(query_terms, doc_terms, V, eps=0.1):
    """
    Computes the log-probability of each document given the query terms
    using the query likelihood language model with Lidstone correction.
    
    Parameters:
        query_terms (list): a list of query terms
        doc_terms (list): a list of terms from a document
        V (int): the size of the vocabulary
        eps (float): the value of epsilon for Lidstone correction
    
    Returns:
        float: the sum of the log-probabilities of each query term given the document
    """
    log_prob = 0
    doc_freqs = Counter(doc_terms)
    doc_len = len(doc_terms)
    for term in query_terms:
        tf = doc_freqs.get(term, 0)
        p = (tf + eps) / (doc_len + eps * V)
        log_prob += math.log(p)
    return log_prob


def compute_doc_probs_dirichlet(query_terms, doc_terms, V, mu=50):
    """
    Computes the log-probability of each document given the query terms
    using the query likelihood language model with Dirichlet smoothing.
    
    Parameters:
        query_terms (list): a list of query terms
        doc_terms (list): a list of terms from a document
        V (int): the size of the vocabulary
        mu (float): the value of mu for Dirichlet smoothing
    
    Returns:
        float: the sum of the log-probabilities of each query term given the document
    """
    log_prob = 0
    doc_freqs = Counter(doc_terms)
    doc_len = len(doc_terms)
    for term in query_terms:
        tf = doc_freqs.get(term, 0)
        p = (tf + mu * V / doc_len) / (doc_len + mu)
        log_prob += math.log(p)
    return log_prob


import csv
from collections import Counter
import math


# Step 6: Retrieve top 100 passages for each query using query likelihood language models

def generate_lm_results(inverted_index, doc_info, query_file, passage_file, output_file, V, smoothing):
    """
    Generates the results for a given language model with the given smoothing method
    
    Parameters:
        inverted_index (dict): the inverted index
        doc_info (dict): a dictionary mapping document IDs to document information
        query_file (str): the path to the file containing the queries
        passage_file (str): the path to the file containing the candidate passages
        output_file (str): the path to the output file to write results to
        V (int): the size of the vocabulary
        smoothing (str): the smoothing method to use ('laplace', 'lidstone', or 'dirichlet')
    """
    reader_queries = csv.reader(open(query_file, 'r', encoding='utf-8'), delimiter='\t')
    queries = [(qid, query) for qid, query in reader_queries]

    reader_passages = csv.reader(open(passage_file, 'r', encoding='utf-8'), delimiter='\t')

    writer = csv.writer(open(output_file, 'w', encoding='utf-8', newline=''), delimiter=',')
    all_scores = []

    for qid, query in queries:
        query_terms = query.lower().split()
        scores = []

        for pid, (doc_terms, doc_len) in doc_info.items():
            if pid in scores:
                continue
            if smoothing == 'laplace':
                log_prob = compute_doc_probs_laplace(query_terms, doc_terms, V)
            elif smoothing == 'lidstone':
                log_prob = compute_doc_probs_lidstone(query_terms, doc_terms, V, eps=0.1)
            elif smoothing == 'dirichlet':
                log_prob = compute_doc_probs_dirichlet(query_terms, doc_terms, V, mu=50)
            else:
                raise ValueError('Invalid smoothing method')

            scores.append((pid, log_prob))

        # Sort the scores for the current query and keep the top 100 passages
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:100]

        # Add the scores to the all_scores list
        all_scores += [(qid, pid, score) for pid, score in scores]

    # Write the results to the output file
    writer.writerows(all_scores)

    return


# Set up variables and call the functions to generate the results for each language model
term_counts = count_terms('passage-collection.txt')
inverted_index, doc_info = build_inverted_index('candidate-passages-top1000.tsv', term_counts)
V = len(inverted_index)
generate_lm_results(inverted_index, doc_info, 'test-queries.tsv', 'candidate-passages-top1000.tsv', 'laplace.csv', V, 'laplace')
generate_lm_results(inverted_index, doc_info, 'test-queries.tsv', 'candidate-passages-top1000.tsv', 'lidstone.csv', V, 'lidstone')
generate_lm_results(inverted_index, doc_info, 'test-queries.tsv', 'candidate-passages-top1000.tsv', 'dirichlet.csv', V, 'dirichlet')
