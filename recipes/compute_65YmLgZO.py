# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
collected_data = dataiku.Dataset("collected_data")
collected_data_df = collected_data.get_dataframe()




# Write recipe outputs
eda_results = dataiku.Folder("65YmLgZO")
eda_results_info = eda_results.get_info()
