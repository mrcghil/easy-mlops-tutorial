# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
model_versions = dataiku.Folder("cjQ040KC")
model_versions_info = model_versions.get_info()


action_df = pd.DataFrame()


# Write recipe outputs
action = dataiku.Dataset("action")
action.write_with_schema(action_df)
