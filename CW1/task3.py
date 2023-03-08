import csv
import math
from collections import defaultdict, Counter
from operator import itemgetter
from task1 import count_terms
from task2 import build_inverted_index


########################## D4 ################################
term_counts = count_terms('passage-collection.txt')
 
inverted_index, doc_info = build_inverted_index('candidate-passages-top1000.tsv', term_counts)

# Step 1: Compute IDF values for each term in the collection
def compute_idfs(inverted_index, doc_info):
    N = len(doc_info) # total number of documents in the collection
    idfs = {} # dictionary to store IDF values for each term
    for term, posting in inverted_index.items():
        df = len(posting) # the number of documents that contain the term
        idf = math.log(N/df)  # compute IDF value using formula
        idfs[term] = idf
    # Remove terms with IDF value of 0
    for term, idf in list(idfs.items()):
        if idf == 0:
            del idfs[term]
    return idfs



# Step 2: Compute TF-IDF vectors for each document
def compute_tfidfs(inverted_index, doc_info, idfs):
    tfidfs = {}
    for pid, (terms, doc_len) in doc_info.items():
        tfidfs[pid] = {}
        term_freqs = Counter(terms) # count the frequency of each term in the list
        for term, freq in term_freqs.items(): # the number of times the term appears in the document
            if term not in idfs:
                continue  # skip terms not in IDFs
            tf = freq/doc_len 
            tfidfs[pid][term] = tf * idfs[term]
    return tfidfs



# Step 3: Compute TF-IDF vectors for each query
with open('test-queries.tsv', 'r', encoding='utf-8') as f_queries:
        reader = csv.reader(f_queries, delimiter='\t')
        queries = [(row[0], row[1]) for row in reader]
def compute_query_tfidfs(queries, idfs):
    query_tfidfs = {}
    for qid, query in queries:
        terms = query.lower().split()
        term_freqs = defaultdict(int)
        for term in terms:
            term_freqs[term] += 1
        query_tfidfs[qid] = {term: freq/len(terms)*idfs[term] for term, freq in term_freqs.items() if term in idfs}
    return query_tfidfs



# Step 4: Compute cosine similarity between queries and documents
def compute_cosine_similarity(inverted_index, doc_info):
    idfs = compute_idfs(inverted_index, doc_info)
    tfidfs = compute_tfidfs(inverted_index, doc_info, idfs)

    with open('test-queries.tsv', 'r', encoding='utf-8') as f_queries, \
         open('candidate-passages-top1000.tsv', 'r', encoding='utf-8') as f_passages, \
         open('tfidf.csv', 'w', encoding='utf-8', newline='') as outfile:
        
        # Read in the queries and store them in a list
        queries = []
        reader_queries = csv.reader(f_queries, delimiter='\t')
        for row in reader_queries:
            queries.append((row[0], row[1]))

        reader_passages = csv.reader(f_passages, delimiter='\t')
        writer = csv.writer(outfile, delimiter=',')

        all_scores = [] # list to hold all scores for each query
        seen_pairs = set()  # keep track of already written (qid, pid) pairs
        
        # Iterate over the queries in the order they appear in the file
        for qid, query in queries:
            query_tfidfs = compute_query_tfidfs([(qid, query)], idfs)
            query_vec = query_tfidfs[qid] # retrieves the TF-IDF vector representation for the current query

            scores = []
            processed_pids = set()  # keep track of pids that have already been processed
            
            # Iterate over the passages and compute cosine similarity scores
            for row in reader_passages:
                pid = row[0]
                if pid in processed_pids:
                    continue
                processed_pids.add(pid)
                doc_vec = tfidfs[pid]
                score = sum(query_vec.get(term, 0) * doc_vec.get(term, 0) for term in set(query_vec.keys()) & set(doc_vec.keys()))
                if (qid, pid) in seen_pairs:
                    continue
                scores.append((qid, pid, score)) # add qid to tuple
                seen_pairs.add((qid, pid))
                if len(scores) >= 100:
                    break
            
            all_scores += scores # add scores to all_scores list
            f_passages.seek(0) # reset file pointer to beginning for next query
        
        # Step 5: Rank documents based on cosine similarity to query and return top 100 for each query
        # Sort all_scores by score value
        all_scores = sorted(all_scores, key=lambda x: (x[0], x[2]), reverse=True) # rank qid then rank score
        top_100_docs = {}
        for qid, pid, score in all_scores:
            if qid not in top_100_docs:
                top_100_docs[qid] = []
            if len(top_100_docs[qid]) >= 100:
                continue
            top_100_docs[qid].append((pid, score))
        
        # Sort the documents for each query by similarity score
        for qid in top_100_docs:
            top_100_docs[qid] = sorted(top_100_docs[qid], key=lambda x: x[1], reverse=True)
        
        # Return the results in the order of the original queries
        results = []
        for qid, _ in queries:
            if qid in top_100_docs:
                results.extend([(qid, pid, score) for pid, score in top_100_docs[qid]])
        writer.writerows(results)

    return

compute_cosine_similarity(inverted_index, doc_info)

########################## D5 ################################

# Step 1: Count the number of times each term appears in each document
# Step 2: Build the inverted index and document info
inverted_index, doc_info = build_inverted_index('candidate-passages-top1000.tsv', term_counts)
def compute_bm25_scores(inverted_index, doc_info, k1=1.2, k2=100, b=0.75):
    # Compute IDF values for each term in the inverted index
    idfs = {}
    N = len(doc_info)
    for term in inverted_index:
        n = len(inverted_index[term])
        idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
        idfs[term] = idf

    # Compute average document length
    total_len = sum(doc_info[pid][1] for pid in doc_info)
    avgdl = total_len / len(doc_info)

    with open('test-queries.tsv', 'r', encoding='utf-8') as f_queries, \
         open('candidate-passages-top1000.tsv', 'r', encoding='utf-8') as f_passages, \
         open('bm25.csv', 'w', encoding='utf-8', newline='') as outfile:

        # Read in the queries and store them in a list
        queries = []
        reader_queries = csv.reader(f_queries, delimiter='\t')
        for row in reader_queries:
            queries.append((row[0], row[1]))

        reader_passages = csv.reader(f_passages, delimiter='\t')
        writer = csv.writer(outfile, delimiter=',')

        all_scores = [] # list to hold all scores for each query
        seen_pairs = set()  # keep track of already written (qid, pid) pairs

        # Iterate over the queries in the order they appear in the file
        for qid, query in queries:
            query_terms = query.lower().split()
            query_freqs = Counter(query_terms)
            query_length = sum(query_freqs.values())
            scores = []
            processed_pids = set()  # keep track of pids that have already been processed

            # Iterate over the passages and compute BM25 scores
            for row in reader_passages:
                pid = row[0]
                if pid in processed_pids:
                    continue
                processed_pids.add(pid)
                doc_terms = row[2].lower().split()
                doc_len = len(doc_terms)
                doc_score = 0
                for term, freq in query_freqs.items():
                    if term not in idfs:
                        continue
                    tf = doc_terms.count(term) / doc_len   
                    tfq = query_freqs[term] / query_length
                    assert tfq != 0 ,'tfq'    
                    idf = idfs[term]
                    numerator = idf * tf * (k1 + 1)  
                    if tf == 0:
                        denominator = k1 * ((1 - b) + b * (total_len / avgdl))
                    else:
                        denominator = tf + k1 * ((1 - b) + b * (total_len / avgdl))
                    bm25_score = (numerator / denominator) * (k2 + 1) * tfq / (k2 + tfq)
                    
                    doc_score += bm25_score
                scores.append((pid, doc_score))
                if len(scores) >= 100:
                    break
                
            # Sort scores by BM25 score value
            scores = sorted(scores, key=itemgetter(1), reverse=True)[:100]
            for pid, score in scores:
                if (qid, pid) in seen_pairs:
                    continue
                writer.writerow([qid, pid, score])
                seen_pairs.add((qid, pid))

            # Reset file pointer to beginning for next query
            f_passages.seek(0)
        
    return
compute_bm25_scores(inverted_index, doc_info, k1=1.2, k2=100, b=0.75)