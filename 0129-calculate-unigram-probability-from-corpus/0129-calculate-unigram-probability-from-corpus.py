def unigram_probability(corpus: str, word: str) -> float:
    # tokens_list = list(corpus.split(" "))
    # total_tokens = len(tokens_list)
    # word_cnt = sum(1 for token in tokens_list if token == word)
    # return round(word_cnt / total_tokens, 4)
    
    # tokens = corpus.split()
    # return round(tokens.count(word) / len(tokens), 4)

    tokens = corpus.split()
    return round(sum(token == word for token in tokens) / len(tokens), 4)

# TC: O(n)
# SC: O(n)