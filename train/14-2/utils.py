import os
import numpy as np

from sklearn.metrics import confusion_matrix

def get_file_list(file_path):

    file_list = [os.path.join(file_path, filename) for filename in os.listdir(file_path)]

    return  file_list

def round_to_nearest_valid_score(score):

    valid_scores = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    return min(valid_scores, key = lambda x: abs(x - score))

def top_k_score(y_true, y_pred, k):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    correct = 0
    
    for true, pred in zip(y_true, y_pred):
        if k == 1:
            if true == pred:
                correct += 1

        elif k == 2:
            if abs(true - pred) <= 0.5:
                correct += 1

        elif k == 3:
            if abs(true - pred) <= 1:
                correct += 1
    
    return correct / len(y_true)