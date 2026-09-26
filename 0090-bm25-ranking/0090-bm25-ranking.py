import numpy as np

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus:
        return []
    N = len(corpus)
    avg_doc_len = sum(len(doc) for doc in corpus) / N
    idf_dict = {}
    for word in set(query):
        df = sum(1 for doc in corpus if word in doc)
        idf_dict[word] = np.log((N + 1) / (df + 1)).item()
    
	scores = []
    for doc in corpus:
        doc_score = 0
        for word in set(query):
            tf = doc.count(word) #/ len(doc) if doc else 0
            bm25 = idf_dict[word] * (
                (tf * (k1 + 1))
                / (tf + k1 * (1 - b + b * (len(doc) / avg_doc_len)))
            )
            doc_score += bm25
        scores.append(round(doc_score, 3))
    return scores