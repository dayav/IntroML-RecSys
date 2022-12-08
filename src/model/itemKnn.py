#Implémentation du itemKnn
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from tqdm import tqdm

class ItemKnn:
    def __init__(self, k, max_item_id):
        self._k = k
        self._max_item_id = max_item_id
        self._session_item = None
        self._item_session = None
        self._sim_matrix = None
    
    def fit(self,data):
        unique_ids = pd.unique(data["SessionId"])
        self.map=dict()
        for idx, ids in enumerate(unique_ids):
            self.map[ids]=idx
        self._session_item = np.zeros((len(unique_ids), self._max_item_id))
        self._item_session = dict()
        for session_id in tqdm(unique_ids, disable=False):
            idx=self.map[session_id]
            items_by_session = (data.loc[data["SessionId"] == session_id])[
                "ItemId"
            ].to_numpy(copy=True)
            items_by_session = np.reshape(items_by_session, (1, -1))
            self._session_item[idx][items_by_session] = 1

            for prod in list(items_by_session[0]):
                if prod in self._item_session:
                    self._item_session[prod].add(session_id)
                else:
                    self._item_session[prod] = {session_id}
        self._session_item = csr_matrix(self._session_item)
        self._sim_matrix = cosine_similarity(self._session_item.T)

    def predict(self, data):
        batch = []
        context = dict()
        unique_ids = pd.unique(data["SessionId"])
        
        for session_id in tqdm(unique_ids, disable=False):
            items_by_session=(data.loc[data["SessionId"] == session_id])[
                "ItemId"
            ].to_numpy(copy=True)
            items_by_session = np.reshape(items_by_session, (1, -1))
            context[session_id]=items_by_session[0][-1]
            batch.append(np.argpartition(self._sim_matrix[context[session_id]],-self._k)[-self._k:])

        return batch
    
    def score(self, data, target):
        n_good_predictions=0
        y_hat = self.predict(data)
        for idx, y in enumerate(target):
            if y in y_hat[idx]:
                n_good_predictions+=1
        score=n_good_predictions/len(target)
        return score

if __name__=='__main__':
    data_vis = pd.read_csv("./session_rec_sigir_data/prepared/sigir_train_full.txt")
    data_eval = data_vis.loc[data_vis["SessionId"] == 5050]
    print(data_vis.shape)
    data_test = data_vis[100001:101000]
    data_vis = data_vis[:100000]

    max_item = np.max(pd.unique(data_vis["ItemId"]))
    print(max_item)
    my_knn = ItemKnn(5, max_item + 1)
    my_knn.fit(data_vis)
    Y = my_knn.predict(data_vis[50000:50100])
    print(Y)