import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.
    
    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
	# IDF(t)=log(N+1 / df(t)+1)+1
	# Adding +1 inside the fraction prevents division by zero if a term never appears.
	# Adding +1 outside the log ensures IDF remains nonzero.
    if not corpus:
        return []
    N = len(corpus)
    idf_dict = {}
    for word in set(query):
        df = sum(1 for doc in corpus if word in doc)
        idf_dict[word] = np.log((N + 1) / (df + 1)).item() + 1
    
    result = []
    for doc in corpus:
        row = []
        for word in query:
            tf = doc.count(word) / len(doc) if doc else 0
            row.append(round(tf * idf_dict[word], 5))
        result.append(row)
    return result

# TC: O(Q × D × L)
# SC: O(Q + DQ)