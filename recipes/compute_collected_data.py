# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs from reference/test/historical dataset
sample_data = dataiku.Dataset("sample_data")
sample_data_df = sample_data.get_dataframe()

# Drop the columns that cannot be used for modelling
sample_data_df = sample_data_df.drop(columns=['long_trend', 'short_trend', 'noise', 'price_sensitivity', 'score'])
# Add a tag to the data
sample_data_df['source'] = "reference"

# Read inputs from the live dateset (results of model being active)
try:
	results_from_model = dataiku.Dataset("results_from_model")
	results_from_model_df = results_from_model.get_dataframe()
	# Drop the columns that cannot be used for modelling
	results_from_model_df = results_from_model_df.drop(columns=['long_trend', 'short_trend', 'noise', 'price_sensitivity', 'score'])
	# Add a tag to the data
	results_from_model_df['source'] = "live"
except Exception as exc:
	print(exc.args)
	print("Data from live model not available.")
	live_data = False


# Collate the two sets
if live_data:
	collected_data_df = pd.concat([sample_data_df, results_from_model_df])
else:
	collected_data_df = sample_data_df

# Write recipe outputs
collected_data = dataiku.Dataset("collected_data")
collected_data.write_with_schema(collected_data_df)
