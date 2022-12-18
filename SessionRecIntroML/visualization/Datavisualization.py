import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Datavisualization:
    def __init__(self, data_path):
        self._data_frame = pd.read_csv(data_path)

    def plot_product_popularity(self):
        self._data_frame.hist(column=["ItemId"], grid=False, sharex=True)
        plt.show()

    def plot_box_session_item(self):
        self._data_frame["SessionId"].value_counts().plot.box()
        plt.show()


data_vis = Datavisualization("C:/Dev/IntroML/projet/src/data/sigir_train_full.txt")
print(data_vis._data_frame)
data_vis.plot_product_popularity()
data_vis.plot_box_session_item()

data_vis.plot_box_session_item()
