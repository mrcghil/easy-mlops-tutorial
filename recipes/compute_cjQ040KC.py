# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
collected_data = dataiku.Dataset("collected_data")
collected_data_df = collected_data.get_dataframe()

model_versions = dataiku.Folder("cjQ040KC")
model_versions_info = model_versions.get_info()
data_folder = dataiku.get_custom_variables()["batches_path"]

# Problem constants
ALL_IN_COST = 1
INITIAL_PRICE = 1.8

collected_data['cost'] = ALL_IN_COST
collected_data['revenue'] = INITIAL_PRICE + collected_data['price_delta']
collected_data['income'] = collected_data['revenue'] - collected_data['cost']

# We want to find the product discount that maximises our expected income
## We model first the relationship between outcome (transaction occurred) and price
## 1D Logistic regression


# Model definition

# Check the performance of the live model to see how we are performing
live_df = collected_data_df[collected_data_df['source'] == "live"].reset_index()

# Write recipe outputs
