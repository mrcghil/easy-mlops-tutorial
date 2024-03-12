# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
sample_batches = dataiku.Folder("zY3ZURve")
sample_batches_info = sample_batches.get_info()


collected_data_df = pd.DataFrame()


# Write recipe outputs
collected_data = dataiku.Dataset("collected_data")
collected_data.write_with_schema(collected_data_df)
