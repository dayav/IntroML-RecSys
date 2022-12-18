import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix
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
        self._session_ids = None

    def fit(self, data):
        session_groups = data.groupby(["SessionId"])
        unique_ids = pd.unique(data["SessionId"])
        self._session_item = lil_matrix(
            (len(unique_ids), self._max_item_id), dtype=np.ubyte
        )
        self._item_session = dict()
        self._session_ids = np.zeros(unique_ids.shape)
        relative_session_ids = 0

        for name, group in tqdm(session_groups):
            values = group["ItemId"].values
            self._session_item[relative_session_ids, values] = 1
            for prod in values:
                if prod in self._item_session:
                    self._item_session[prod].add(name)
                else:
                    self._item_session[prod] = {name}
            self._session_ids[relative_session_ids] = name
            relative_session_ids += 1

        # for session_id, relative_session_id in tqdm(
        #     zip(unique_ids, relative_session_ids), disable=False
        # ):
        #     t = data.groupby(["SessionId"])

        #     items_by_session = (data.loc[data["SessionId"] == session_id])[
        #         "ItemId"
        #     ].to_numpy(copy=True)
        #     items_by_session = np.reshape(items_by_session, (1, -1))
        #     self._session_item[relative_session_id, items_by_session[0]] = 1
        #     self._item_session_test[items_by_session[0], session_id] = 1

        # for prod in items_by_session[0]:
        #     if prod in self._item_session:
        #         self._item_session[prod].add(session_id)
        #     else:
        #         self._item_session[prod] = {session_id}

        self._session_item = csr_matrix(self._session_item)

    def eval(self, session_item_data):
        session_item = session_item_data["ItemId"].to_numpy(copy=True)
        session_item_bool = np.zeros(self._max_item_id, dtype=np.ubyte)
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

        if len(filtered_candidates) == 0:
            return np.zeros(self._predictions)
        relative_filtered_candidates = np.argwhere(
            np.isin(self._session_ids, filtered_candidates)
        ).flatten()

        candidate_sessions_bool = self._session_item[relative_filtered_candidates]
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
        session_groups = data.groupby(["SessionId"])
        for name, group in tqdm(session_groups):
            rec = self.eval(group)
            if len(rec) < self._predictions:
                zero_len = self._predictions - len(rec)
                zero_pad = np.zeros(zero_len)
                rec = np.hstack((rec, zero_pad))
            batch.append(rec)
        # for i, session in tqdm(data.iterrows(), total=data.shape[0]):
        #     if previous_item == session["SessionId"]:
        #         continue
        #     rec = self.eval(data.loc[data["SessionId"] == session["SessionId"]])
        #     if len(rec) < self._predictions:
        #         zero_len = self._predictions - len(rec)
        #         zero_pad = np.zeros(zero_len)
        #         rec = np.hstack((rec, zero_pad))
        #     previous_item = session["SessionId"]

        #     batch.append(rec)
        return np.array(batch)

    def score(self, data, real_data):
        # we assume we have one real value
        data_pred = self.predict(data)
        well_predicted = 0
        all_predictions = 0
        for real, pred in zip(real_data, data_pred):
            real_found_in_pred = np.isin(real, pred, assume_unique=True)
            all_predictions += 1
            if real_found_in_pred:
                well_predicted += 1

        return well_predicted / all_predictions

    def mrr_score(self, data, target):
        """
        Compute the mean reciprocal error of predictions over target.
        """
        mrr_sum = 0
        y_hat = self.predict(data)
        for idx, y in enumerate(target):
            for i in range(1, self._k + 1):
                if self._predictions >= i and y == y_hat[idx][-i]:
                    mrr_sum += 1 / i
        mrr_score = mrr_sum / len(target)
        return mrr_score
