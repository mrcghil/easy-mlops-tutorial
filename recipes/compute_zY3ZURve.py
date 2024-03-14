# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import datetime
import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Connect to folder
sample_batches = dataiku.Folder("zY3ZURve")
sample_batches_info = sample_batches.get_info()

# Path in folder
data_folder = dataiku.get_custom_variables()["batches_path"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Model
# seed = 112233
# np.random.seed(seed)
reference_time = datetime.datetime(2024, 1, 1)

def get_features(current_time: datetime.datetime, fixed_delta = "random", size:tuple = (1000,)) -> pd.DataFrame:
    # Time
    time_delta = current_time - reference_time
    timestamps = pd.date_range(current_time, periods=size[0], freq="s")

    # Trend increses in time to saturation
    long_trend = min((0.008 * time_delta.total_seconds() / (24 * 60 * 60)), 0.80) * np.ones(size)

    # Intraday
    short_trend = 0.1 * np.sin( 2 * np.pi * (time_delta.seconds / 3600) / 24 ) * np.ones(size)

    # Price delta
    if type(fixed_delta) == str and fixed_delta == "random":
        price_delta = np.random.uniform(-1, 1, size)
    elif type(fixed_delta) == int or type(fixed_delta) == float:
        price_delta = fixed_delta * np.ones(size)

    # Random noise
    noise_a = np.random.normal(0, 0.3, size)

    df = pd.DataFrame(
        {
            "timestamps": timestamps,
            "long_trend": long_trend.tolist(),
            "short_trend": short_trend.tolist(),
            "noise": noise_a.tolist(),
            "price_delta": price_delta.tolist(),
        }
    )

    return df

def get_outcome(features: pd.DataFrame) -> pd.DataFrame:
    # Price dependency
    features['price_sensitivity'] = - 3 * ((1 / (1 + np.exp(-features['price_delta']))) - 0.5)
    features['score'] = features['long_trend'] + features['short_trend'] + features['price_sensitivity'] + features['noise']
    features['output'] = features['score'] >= 0.7
    return features

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Produce the data and save to the folder
current_time = datetime.datetime.now()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Produce features
features = get_features(current_time, "random", (1000,))
print(features.head(5))
features = get_outcome(features)
print(features.describe())

# Drop this later to show
# features = features.drop(columns=["long_trend", "short_trend", "noise"])
# print(features.head(20))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save data to the folder
with sample_batches.get_writer(os.path.join(data_folder, "_".join(["data", current_time.strftime("%Y%m%d_%H_%M_%S") + ".csv"]))) as w:
    w.write(
        features.to_csv().encode()
    )

# Connect to dataset
sample_data_dataset = dataiku.Dataset("sample_data")
sample_data_dataset.write_with_schema(features)