import pandas as pd
from model import Knn
from data import DataManipulation
import numpy as np

data_manip = DataManipulation()
data_vis = pd.read_csv("C:/Dev/IntroML/projet/src/data/sigir_train_full.txt")
data_eval = data_vis.loc[data_vis["SessionId"] == 5050]
print(data_vis.shape)
data_vis = data_vis[:100000]


data_eval = data_vis[0:6]
previous_item = None
print(f"data_eval before: {data_eval}")
data_eval, real_value = data_manip.prepare_data_for_test(data_eval)
print(f"data_eval after : {data_eval}")

max_item = np.max(pd.unique(data_vis["ItemId"]))
print(max_item)
my_knn = Knn(5, max_item + 1, 10)
my_knn.fit(data_vis)
Y = my_knn.predict(data_eval)
my_knn.score(data_eval, real_value)
print(Y)
