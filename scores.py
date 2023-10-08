import numpy as np


def get_final_scores(test_scores, test_score):
    test_scores = np.column_stack((test_scores, test_score))
    return test_scores.max(axis=1)