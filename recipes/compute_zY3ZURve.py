# -*- coding: utf-8 -*-
import dataiku
import datetime
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Connect to folder
sample_batches = dataiku.Folder("zY3ZURve")
sample_batches_info = sample_batches.get_info()


