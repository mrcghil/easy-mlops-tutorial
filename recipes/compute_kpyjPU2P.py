# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
action = dataiku.Dataset("action")
action_df = action.get_dataframe()




# Write recipe outputs
sample_results = dataiku.Folder("kpyjPU2P")
sample_results_info = sample_results.get_info()
