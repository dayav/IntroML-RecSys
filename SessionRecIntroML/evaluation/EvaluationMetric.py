import numpy as np


def hit_rate(predicted_items: np.ndarray, real_items: np.ndarray, k) -> float:
    hits = 0
    for real, pred in zip(real_items, predicted_items):

        if real in pred[:k]:
            hits += 1

    hit_rate = hits / len(predicted_items)
    return hit_rate


def mean_reciprocal_rank(predicted_items: np.ndarray, real_items: list) -> float:
    idx = 0
    # We assume the predicted_item is sorted by their rank
    mrratk_score = 0
    for real, pred in zip(real_items, predicted_items):

        if real in pred:

            rank_idx = np.argwhere(np.array(pred) == real)[0][0] + 1
            # score higher-ranked ground truth higher than lower-ranked ground truth
            mrratk_score += 1 / rank_idx

    mrratk_score /= len(real_items)

    return mrratk_score
