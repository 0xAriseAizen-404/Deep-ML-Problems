import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    # Your code here
    probabilities = []
    mse = 0
    for feature, label in zip(features, labels):
        z = sum(w * x for w, x in zip(weights, feature)) + bias
        logit = round(1 / (1 + math.exp(-z)), 4)
        probabilities.append(logit)
        mse = ((mse * (len(probabilities) - 1)) + (label - logit) ** 2) / len(probabilities)
    return probabilities, round(mse, 4)
    
        # mse += (label - logit) ** 2
    # return probabilities, round(mse/len(probabilities), 4)