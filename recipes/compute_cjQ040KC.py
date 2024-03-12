# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
collected_data = dataiku.Dataset("collected_data")
collected_data_df = collected_data.get_dataframe()




# Write recipe outputs
model_versions = dataiku.Folder("cjQ040KC")
model_versions_info = model_versions.get_info()
