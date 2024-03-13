# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
import datetime
import json
import os
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
collected_data = dataiku.Dataset("collected_data")
collected_data_df = collected_data.get_dataframe()

model_versions = dataiku.Folder("cjQ040KC")
model_versions_info = model_versions.get_info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Problem constants
ALL_IN_COST = 1
INITIAL_PRICE = 1.8

def income_function_single(price_delta:np.ndarray) -> np.ndarray:
    return (INITIAL_PRICE + price_delta) - ALL_IN_COST

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Where to save successive modelling tests
class ModelParameters:

    def __init__(self, hypers, parameters, outputs, scores):
        # Hyperparams
        self.hypers = hypers
        # Relevant paramters
        self.parameters = parameters
        # outputs
        self.outputs = outputs
        # self score
        self.scores = scores
		# timestamp
		self.timestamp = datetime.datetime.now()

    def to_json(self):
        return json.dumps({
            "hypers": self.hypers,
            "parameters": self.parameters,
            "outputs": self.outputs,
            "scores":self.scores
			"timestamp": self.timestamp.strftime("%Y%m%d_%H_%M_%S")
        })

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check the performance of the live model to see how we are performing
live_df = collected_data_df[collected_data_df['source'] == "live"].reset_index()

# If the performance is good we can skip the model creation below
if True:
    # Model
    REFIT_MODEL = True
else:
    REFIT_MODEL = False

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# We want to find the product discount that maximises our expected income

## This will contain all the produced models
models_collection = []

## Iterate for all the model builds (data included)
if REFIT_MODEL:
    for index in range(1):
        print(f"Running tuning: {index+1} ...")

        ## Can select the data better (perform split and ...)
        X = collected_data_df['price_delta'].values.reshape(-1, 1)
        Y = collected_data_df['output'].values

        ## 1D Logistic regression
        ## We model first the relationship between outcome (transaction occurred) and price
        purchase_classifier = LogisticRegression(
            random_state=None,
            # solver='lbfgs',
            # penalty='l2'
            # max_iter=100,
        ).fit(
            X,
            Y,
        )

        ## This is the simplest thing not the best
        simple_fit_score = purchase_classifier.score(X,Y)

        ## Calculate the optimal price for the period
        nodes = np.linspace(-1,1,1001)
        ### Probability of purchase
        purchase_prob = purchase_classifier.predict_proba(nodes.reshape(-1, 1))[:,1]
        purchase_pdf = purchase_prob / np.trapz(purchase_prob, nodes)
        ### Incomes
        income_line = income_function_single(nodes)

        optimal_price_delta_ind = np.argmax(income_line * purchase_pdf)
        optimal_price_delta = nodes[optimal_price_delta_ind]
        print(f"Optimal price delta at {optimal_price_delta} ...")

        plt.plot(nodes, income_line)
        plt.plot(nodes, purchase_pdf)
        plt.plot(nodes, income_line * purchase_pdf)
        plt.show()

        models_collection.append(
            ModelParameters(
                {},
                {},
                {
                    "action": optimal_price_delta
                },
                {
                    "score": simple_fit_score
                }
            )
        )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs

if REFIT_MODEL:
    ## Write latest model card
    current_time = datetime.datetime.now()
    latest_model = {
        "path": current_time.strftime("%Y%m%d_%H_%M_%S")
    }
    with model_versions.get_writer("latest_model.json") as w:
            w.write(
                json.dumps(latest_model).encode()
            )
    ##
    for ind, model in enumerate(models_collection):
        with model_versions.get_writer(os.path.join(latest_model['path'], "_".join(["model", f"{ind}.json"]))) as w:
            w.write(
                model.to_json().encode()
            )