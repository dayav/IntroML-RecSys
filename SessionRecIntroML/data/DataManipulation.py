from pathlib import Path
import pickle
import numpy as np

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

TRAIN_RATIO = 0.5

TRAIN_PATH = Path("./src/data/browsing_train.csv")
# TEST_PATH = Path("baselines/session_rec_sigir_data/test/rec_test_sample.json")
SessionId = "SessionId"
ItemId = "ItemId"
Time = "Time"

PREPARED_FOLDER = Path("./session_rec_sigir_data/prepared")
PREPARED_FOLDER.mkdir(parents=True, exist_ok=True)

PREPARED_TRAIN_PATH = PREPARED_FOLDER / "sigir_train_full.txt"
PREPARED_TEST_PATH = PREPARED_FOLDER / "sigir_test.txt"
ITEM_LABEL_ENCODING_MAP_PATH = PREPARED_FOLDER / "item_label_encoding.p"


class DataManipulation:
    def label_encode_series(self, series: pd.Series):
        """
        Applies label encoding to a Pandas series and returns the encoded series,
        together with the label to index and index to label mappings.

        :param series: input Pandas series
        :return: Pandas series with label encoding, label-integer mapping and integer-label mapping.
        """
        labels = set(series.unique())
        label_to_index = {l: idx for idx, l in enumerate(labels)}
        index_to_label = {v: k for k, v in label_to_index.items()}
        return series.map(label_to_index), label_to_index, index_to_label

    def create_data_train_format(self):
        train_data_df = pd.read_csv(TRAIN_PATH)

        session_ids = set(train_data_df["session_id_hash"].unique())
        train_cutoff = int(len(session_ids) * TRAIN_RATIO)
        train_session_ids = list(session_ids)[:train_cutoff]
        train_data_df = train_data_df[
            train_data_df["session_id_hash"].isin(train_session_ids)
        ]

        # Filter out:
        # * `remove from cart` events to avoid feeding them to session_rec as positive signals
        # * rows with null product_sku_hash
        # * sessions with only one action
        train_data_df = train_data_df[train_data_df["product_action"] != "remove"]
        train_data_df = train_data_df.dropna(subset=["product_sku_hash"])
        train_data_df["session_len_count"] = train_data_df.groupby("session_id_hash")[
            "session_id_hash"
        ].transform("count")
        train_data_df = train_data_df[train_data_df["session_len_count"] >= 2]

        train_data_df = train_data_df[train_data_df["product_action"] != "remove"]
        train_data_df = train_data_df.dropna(subset=["product_sku_hash"])
        train_data_df["session_len_count"] = train_data_df.groupby("session_id_hash")[
            "session_id_hash"
        ].transform("count")
        train_data_df = train_data_df[train_data_df["session_len_count"] >= 2]

        # sort by session, then timestamp
        train_data_df = train_data_df.sort_values(
            ["session_id_hash", "server_timestamp_epoch_ms"], ascending=True
        )

        # Encode labels with integers
        (
            item_id_int_series,
            item_label_to_index,
            item_index_to_label,
        ) = self.label_encode_series(train_data_df.product_sku_hash)
        item_string_set = set(item_label_to_index.keys())

        # Add tokenized session ID, tokenized item ID, and seconds since epoch time.
        train_data_df[SessionId] = train_data_df.groupby(
            [train_data_df.session_id_hash]
        ).grouper.group_info[0]
        train_data_df[Time] = train_data_df.server_timestamp_epoch_ms / 1000
        train_data_df[ItemId] = item_id_int_series

        # Get final dataframe
        final_train_df = train_data_df[[SessionId, ItemId, Time]]

        # Generate CSV and label encoder
        final_train_df.to_csv(PREPARED_TRAIN_PATH, sep=",", index=False)
        pickle.dump(item_index_to_label, ITEM_LABEL_ENCODING_MAP_PATH.open(mode="wb"))
        print("Done generating 'prepared' for training")

    def prepare_data_for_test(self, datas):
        session_groups = datas.groupby(["SessionId"])
        tests_ = (group.iloc[:-1, :] for name, group in tqdm(session_groups))
        test = pd.concat(tests_)
        real_value = [group.iloc[-1, :].ItemId for name, group in tqdm(session_groups)]

        return test, np.array(real_value)

    def train_test_split(self, datas, train_size, test_size):
        unique_session = pd.unique(datas["SessionId"])
        if (train_size is not None) and (train_size > unique_session.shape[0]):
            real_train_size = None
        else:
            real_train_size = train_size
        print(f"train size was modified real train session size is {real_train_size}")
        session_id_train, session_id_test = train_test_split(
            unique_session,
            test_size=test_size,
            train_size=real_train_size,
            shuffle=True,
        )

        return (
            datas[datas.SessionId.isin(session_id_train)],
            datas[datas.SessionId.isin(session_id_test)],
            real_train_size,
        )

    def leave_one_out_split(self, datas):
        unique_session = pd.unique(datas["SessionId"])
        loo = LeaveOneOut()

        for train_index, valid_index in loo.split(unique_session):
            yield (
                datas[datas.SessionId.isin(train_index)],
                datas[datas.SessionId.isin(valid_index)],
            )

    def kFold_split(self, datas, n_split=5):
        unique_session = pd.unique(datas["SessionId"])
        loo = KFold(n_splits=n_split, shuffle=True)

        for train_index, valid_index in loo.split(unique_session):
            yield (
                datas[datas.SessionId.isin(train_index)],
                datas[datas.SessionId.isin(valid_index)],
            )

    def Split(self, datas, n_split=5):
        unique_session = pd.unique(datas["SessionId"])
        index_split = np.array_split(unique_session, n_split)
        for train_index in index_split:
            yield (datas[datas.SessionId.isin(train_index)])
