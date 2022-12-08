import pandas as pd


browsing = pd.read_csv("C:/Dev/IntroML/projet/data/browsing_train.csv")
print(f"column ID : {browsing.columns}")
print(browsing[["session_id_hash", "event_type", "product_action"]])
print(browsing.info())
print(f'unique product action : {browsing["product_action"].unique()}')
print(f'unique section id action : {browsing["product_action"].unique()}')
