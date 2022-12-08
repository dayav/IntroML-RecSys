import json
from pathlib import Path
import pickle

import pandas as pd

TRAIN_RATIO = 0.5

TRAIN_PATH = Path("baselines/session_rec_sigir_data/train/browsing_train_sample.csv")
TEST_PATH = Path("baselines/session_rec_sigir_data/test/rec_test_sample.json")
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
