import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from tqdm import tqdm


class Knn:
    """
    try to implement Session-Based KNN fromm
    When Recurrent Neural Networks meet the Neighborhood for Session-Based Recommendation
    """

    def __init__(self, k, max_item_id, number_prediction):
        self._k = k
        self._max_item_id = max_item_id
        self._session_item = None
        self._item_session = None
        self._predictions = number_prediction

    def fit(self, data):

        unique_ids = pd.unique(data["SessionId"])
        self._session_item = np.zeros((len(unique_ids), self._max_item_id))
        self._item_session = dict()
        for session_id in tqdm(unique_ids, disable=False):

            items_by_session = (data.loc[data["SessionId"] == session_id])[
                "ItemId"
            ].to_numpy(copy=True)
            items_by_session = np.reshape(items_by_session, (1, -1))
            self._session_item[session_id][items_by_session] = 1

            for prod in list(items_by_session[0]):
                if prod in self._item_session:
                    self._item_session[prod].add(session_id)
                else:
                    self._item_session[prod] = {session_id}

        self._session_item = csr_matrix(self._session_item)

    def eval(self, session_item_data):

        session_item = session_item_data["ItemId"].to_numpy(copy=True)
        session_item_bool = np.zeros(self._max_item_id)
        session_item_bool[session_item] = 1
        session_item_bool = np.reshape(session_item_bool, (1, -1))

        candidates = set()
        for itm in session_item:
            current_item_sessions = self._item_session.get(itm)

            if current_item_sessions is None:
                continue
            candidates.update(current_item_sessions)
        candidates = np.fromiter(candidates, dtype=np.int64)

        if self._k >= len(candidates):
            filtered_candidates = candidates
        else:
            random = np.random.choice(
                range(len(candidates)), size=self._k, replace=False
            )
            # Candidates filtered Randomly
            filtered_candidates = candidates[random][: self._k]

        candidate_sessions_bool = self._session_item[filtered_candidates]
        neighbors_similarity = cosine_similarity(
            session_item_bool, candidate_sessions_bool
        )[0]

        candidate_items_scores = candidate_sessions_bool.multiply(
            neighbors_similarity[:, None]
        )
        # np.matrix -> np.array
        candidate_items_ranking = np.asarray(candidate_items_scores.sum(axis=0))[0]

        # Sort recommendations by score, remove 0, invalid & prods already in the session
        recommendations = np.argsort(-candidate_items_ranking)
        topk_rec = recommendations[: self._predictions]

        return topk_rec

    def predict(self, data):
        batch = []
        previous_item = None
        for i, session in data.iterrows():
            if previous_item == session["SessionId"]:
                continue
            rec = self.eval(data.loc[data["SessionId"] == session["SessionId"]])
            if len(rec) < self._predictions:
                zero_len = self._predictions - len(rec)
                zero_pad = np.zeros(zero_len)
                rec = np.hstack((rec, zero_pad))
            previous_item = session["SessionId"]

            batch.append(rec)
        return np.array(batch)

    def score(self, data, real_data):
        #we assume we have one real value
        data_pred = self.predict(data)
        for real, pred in zip(real_data, data_pred):
            real_found_in_pred = np.isin(pred, real, assume_unique=True)
            print(real_found_in_pred)
