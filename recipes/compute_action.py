# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import datetime
import numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
model_versions = dataiku.Folder("cjQ040KC")
model_versions_info = model_versions.get_info()

# Get the latest model build path
with lts_storage.get_download_stream(full_payment_list_file) as f:
	full_payment_list = json.loads(
		f.read().decode()
	)

# Load the model score-cards


# Decide which model is best to run


# Predict the model output


# Pass the model output and other relevant info to the outputs
action_df = pd.DataFrame({
	"timestamp": [pd.Timestamp(year=2024, month=3, day=12)],
	"optimal_price_delta": [-0.9]
})

# Write recipe outputs
action = dataiku.Dataset("action")
action.write_with_schema(action_df)
