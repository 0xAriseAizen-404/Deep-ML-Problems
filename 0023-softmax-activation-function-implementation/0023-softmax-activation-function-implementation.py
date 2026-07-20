import math

def softmax(scores: list[float]) -> list[float]:
    mx = max(scores)
    scores = [x - mx for x in scores]
    exp_scores = [math.exp(x) for x in scores]
    total = sum(exp_scores)
    return [round(x / total, 4) for x in exp_scores]