# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import datetime
import json
import numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
model_versions = dataiku.Folder("cjQ040KC")
model_versions_info = model_versions.get_info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get the latest model build path
with model_versions.get_download_stream('latest_model.json') as f:
    latest_model_info = json.loads(
        f.read().decode()
    )
print(latest_model_info)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Find the model score-cards
model_versions_list = model_versions.list_paths_in_partition()

ALL = False

if ALL:
    model_card_paths = [path for path in model_versions_list if "latest_model.json" not in path ]
else:
    model_card_paths = [path for path in model_versions_list if latest_model_info['path'] in path ]

# Load the model scorecards
model_cards = []
for model_card_path in model_card_paths:
    with model_versions.get_download_stream(model_card_path) as f:
        model_card = json.loads(
            f.read().decode()
        )
    model_cards.append(model_card)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Model cards flattening (could be done better to and recursively once the data are fuller)
actions = [card['outputs']['action'] for card in model_cards]
scores = [card['scores']['score'] for card in model_cards]
timestamps = [card['timestamp'] for card in model_cards]

models_collection = pd.DataFrame({
    "timestamp": timestamps,
    "optimal_price_delta": actions,
    "scores": scores,
})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Decide which model is best to run
models_collection = models_collection.sort_values('scores', ascending=False)

# Write recipe outputs
action = dataiku.Dataset("action")
action.write_with_schema(models_collection)