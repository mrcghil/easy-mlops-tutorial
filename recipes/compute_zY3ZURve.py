# -*- coding: utf-8 -*-
import dataiku
import datetime
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Connect to folder
sample_batches = dataiku.Folder("zY3ZURve")
sample_batches_info = sample_batches.get_info()

# Path in folder
data_folder = dataiku.get_custom_variables()["batches_path"]

# Model
def get_features(current_time: datetime.datetime) -> pd.DataFrame:
	pass

def get_measurement(features: pd.DataFrame) -> pd.Series:
	pass

# Produce the data and save to the folder

