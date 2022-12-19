import pandas as pd
from model import Knn, ItemKnn
from data import DataManipulation
from matplotlib import pyplot
from evaluation.EvaluationMetric import mean_reciprocal_rank
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

data_manip = DataManipulation()
data = pd.read_csv("../sigir_train_full.txt")
data_vis = data
data_eval = data_vis.loc[data_vis["SessionId"] == 5050]
print(data_vis.shape)
data_vis = data_vis[:100000]
train, tests, _ = data_manip.train_test_split(data, None, 0.25)

# ---------Test models here-----------------
# data_eval = data_vis[0:100]
# previous_item = None
# print(f"data_eval before: {data_eval}")
# data_eval, real_value = data_manip.prepare_data_for_test(data_eval)
# print(f"data_eval after : {data_eval}")

max_item = np.max(pd.unique(data["ItemId"]))
# print(max_item)
# my_knn = Knn(5, max_item + 1, 10)
# my_knn.fit(data_vis)
# Y = my_knn.predict(data_eval)
# my_knn.score(data_eval, real_value)
# print(Y)

# item_knn = ItemKnn(5, max_item + 1)
# item_knn.fit(data_vis)
# Y = item_knn.predict(data_eval)
# accuracy = item_knn.score(data_eval, real_value)
# print(accuracy)

# ----------Parameters tuning-----------------
# Les parametres importants sont le nombre de voisin a
# considerer et la valeur de k. La valeur de k doit rester réaliste
# dans le sens ou on prédit des articles.

# predictors = [Knn(5, max_item + 1, 100)]
# training_size = [10000, 200000, 500000, 1000000]
# training_size = [100000]
# accuracy = dict()
# mrr_accuracy = dict()
# k_values = [3, 5, 10, 15, 20, 30, 50, 100]
# test_size = 10000

# for prd in predictors:
#     for train_size in training_size:
#         # train_set, test_set = train_test_split(data,test_size=test_size,train_size=train_size)
#         accuracy[train_size] = []
#         mrr_accuracy[train_size] = []
#         train_set = data[:train_size]
#         test_set = data[train_size : train_size + test_size]
#         max_item = np.max(pd.unique(train_set["ItemId"]))
#         prd._max_item_id = max_item + 1
#         prd.fit(train_set)
#         test_set, test_target = data_manip.prepare_data_for_test(test_set)
#         for k in k_values:
#             prd._k = k
#             # accuracy[train_size].append(prd.score(test_set, test_target))
#             predicted = prd.predict(test_set)
#             mrr_accuracy[train_size].append(
#                 mean_reciprocal_rank(predicted, test_target)
#             )


# pyplot.figure()
# for size in training_size:
#     pyplot.plot(k_values, mrr_accuracy[size], label=f"t_size={size}")
# pyplot.xlabel("k")
# pyplot.ylabel("Accuracy")
# pyplot.legend()
# pyplot.show()
def construct_session_sequences(df, sessionID, itemID):

    session_groups = df.groupby([sessionID])
    session_seq = []
    unique_ids = pd.unique(data["SessionId"])
    session_item = lil_matrix((len(unique_ids), max_item + 1), dtype=np.ubyte)
    for name, group in tqdm(session_groups):
        session_item[name, group[itemID].values] = 1

    return session_item


# sessions = construct_session_sequences(train, "SessionId", "ItemId")
# print(f"full test size : {tests.shape}")
# for train_d in data_manip.Split(tests, 5):
#     print(f"train_d size : {train_d.shape}")

# test_set, test_target = data_manip.prepare_data_for_test(train_t)
# valid_acc += knn.mrr_score(test_set, test_target)
# count += 1

predictors = [Knn(5, max_item + 1, 100)]
training_size = [10000, 200000, 500000, 1000000]
accuracy = dict()
mrr_accuracy = dict()
k_values = [3, 5, 10, 15, 20, 30, 50, 100]
test_size = 10000

k_values = [3, 5, 10, 15, 20, 30, 50, 100]
data_train_fine, _, _ = data_manip.train_test_split(train, 80000, None)
for k in k_values:

    knn = ItemKnn(k, max_item + 1)

    count = 0
    train_acc, valid_acc = 0, 0
    for train_d, train_t in data_manip.kFold_split(data_train_fine, 5):
        print(f"train_d size : {train_d.shape}")
        print(f"train_t size : {train_t.shape}")
        knn.fit(train_d)
        # print(f"eval_d size : {eval_d.shape}")
        # print(f"eval_t size : {eval_t.shape}")

        test_set, test_target = data_manip.prepare_data_for_test(train_t)
        score = knn.mrr_score(test_set, test_target, 3)
        y_hat = knn.predict(test_set)
        valid_acc += mean_reciprocal_rank(y_hat, test_target, 5)
        count += 1

    # if weight == "uniform":
    #     scoresUniformWeights.append(valid_acc / count)
    # if weight == "distance":
    #     scoresDistanceWeights.append(valid_acc / count)

for prd in predictors:
    for train_size in training_size:
        print(train.shape)
        train_set, test_set, train_size = data_manip.train_test_split(
            train, train_size, test_size
        )

        mrr_accuracy[train_size] = []

        max_item = np.max(pd.unique(train_set["ItemId"]))
        print(max_item)
        prd._max_item_id = max_item + 1
        prd.fit(train_set)
        test_set, test_target = data_manip.prepare_data_for_test(test_set)
        for k in k_values:
            prd._k = k
            predicted = prd.predict(test_set)
            mrr_accuracy[train_size].append(
                mean_reciprocal_rank(predicted, test_target)
            )


pyplot.figure()
for size in training_size:
    pyplot.plot(k_values, mrr_accuracy[size], label=f"t_size={size}")
pyplot.xlabel("k")
pyplot.ylabel("Accuracy")
pyplot.legend()
pyplot.show()
