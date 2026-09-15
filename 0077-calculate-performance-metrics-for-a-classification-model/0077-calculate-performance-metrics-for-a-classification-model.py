def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    # Implement your code here
    TP, FN, FP, TN = 0, 0, 0, 0
    for x, y in zip(actual, predicted):
        if x == 1 and y == 1:
            TP += 1
        elif x == 1 and y == 0:
            FN += 1
        elif x == 0 and y == 1:
            FP += 1
        else:
            TN += 1
    
    confusion_matrix = [[TP, FN], [FP, TN]]
    
    if TP + FN + FP + TN > 0.0:
        accuracy = (TP + TN) / (TP + FN + FP + TN)
    else:
        accuracy = 0.0
    
    if TP + FP > 0.0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0
    if TP + FN > 0.0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    
    if FP + TN > 0.0:
        specificity = TN / (FP + TN)
    else:
        sepcificity = 0.0
        
    if FN + TN > 0.0:
        negativePredictive = TN / (TN + FN)
    else:
        negativePredictive = 0.0

    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)

# TC: O(n)
# SC: O(1)