# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import datetime
import numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
model_versions = dataiku.Folder("cjQ040KC")
model_versions_info = model_versions.get_info()


action_df = pd.DataFrame({
	"timestamp": [pd.Timestamp(year=2024, month=3, day=12)],
	"optimal_price_delta": [-0.9]
})


# Write recipe outputs
action = dataiku.Dataset("action")
action.write_with_schema(action_df)
