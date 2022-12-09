import pandas as pd
from model import Knn, ItemKnn
from data import DataManipulation
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np

data_manip = DataManipulation()
data=pd.read_csv("./session_rec_sigir_data/prepared/sigir_train_full.txt")
data_vis = data
data_eval = data_vis.loc[data_vis["SessionId"] == 5050]
print(data_vis.shape)
data_vis = data_vis[:100000]

#---------Test models here-----------------
data_eval = data_vis[0:100]
previous_item = None
print(f"data_eval before: {data_eval}")
data_eval, real_value = data_manip.prepare_data_for_test(data_eval)
print(f"data_eval after : {data_eval}")

max_item = np.max(pd.unique(data_vis["ItemId"]))
print(max_item)
# my_knn = Knn(5, max_item + 1, 10)
# my_knn.fit(data_vis)
# Y = my_knn.predict(data_eval)
# my_knn.score(data_eval, real_value)
# print(Y)

item_knn = ItemKnn.ItemKnn(5, max_item + 1)
item_knn.fit(data_vis)
Y=item_knn.predict(data_eval)
accuracy=item_knn.score(data_eval, real_value)
print(accuracy)

#----------Parameters tuning-----------------
#Les parametres importants sont le nombre de voisin a
#considerer et la valeur de k. La valeur de k doit rester réaliste
#dans le sens ou on prédit des articles.

predictors=[ItemKnn.ItemKnn(5, max_item + 1)]
training_size=[10000,200000,500000,1000000]
accuracy=dict()
mrr_accuracy=dict()
k_values=[3,5,10,15,20,30,50,100]
test_size=10000

for prd in predictors:
    for train_size in training_size:
        #train_set, test_set = train_test_split(data,test_size=test_size,train_size=train_size)
        accuracy[train_size] = []
        mrr_accuracy[train_size] = []
        train_set=data[:train_size]
        test_set=data[train_size:train_size+test_size]
        max_item = np.max(pd.unique(train_set["ItemId"]))
        prd._max_item_id=max_item+1
        prd.fit(train_set)
        test_set, test_target = data_manip.prepare_data_for_test(test_set)
        for k in k_values:
            prd._k=k
            accuracy[train_size].append(prd.score(test_set, test_target))
            mrr_accuracy[train_size].append(prd.mrr_score(test_set, test_target))


pyplot.figure()
for size in training_size:
    pyplot.plot(k_values,mrr_accuracy[size], label=f't_size={size}')
pyplot.xlabel('k')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.show()
test=0





