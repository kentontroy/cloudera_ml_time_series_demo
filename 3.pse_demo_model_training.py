###########################################################################
# @Team:     Cloudera Partner Solutions Engineering
# @Author:   kdavis@cloudera.com
# @Purpose:  Discover the best Auto ARIMA model on the Airlines demo data
#            Serialize the model for reuse in predictive analytics
# @Modified: 11152021 init
###########################################################################

import boto3
import cdsw
import joblib
import json
import numpy as np
import pandas as pd
import pmdarima as pm
import sys
import uuid
from datetime import datetime
from pmdarima.model_selection import train_test_split
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer, LogEndogTransformer

input = json.loads(sys.argv[1])

###########################################################################
# Load, transform, and aggregate the data by the reformatted date values
###########################################################################

df = pd.read_csv("resources/data/airlines/airlines.csv")
df.dropna(inplace=True)

dfAgg = df.groupby(["Time.Label"])["Statistics.Flights.Cancelled"].sum().reset_index()
print(dfAgg.head())

###########################################################################
# Create training and testing data sets by splitting the data
###########################################################################

test = int(input.get("test_set_size", 1))

y_train, y_test = train_test_split(
   dfAgg["Statistics.Flights.Cancelled"], 
   train_size=len(dfAgg["Statistics.Flights.Cancelled"]) - test)

###########################################################################
# Create an ML pipeline
###########################################################################
# Automatically discover the optimal order for an ARIMA model after 
# transforming the data to account for seasonality

# Ref: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
# Auto-ARIMA works by conducting differencing tests to determine the order of seasonal
# and non-seasonal component differencing

# In general:
# ARIMA(p,d,q)(P,D,Q)m
# p is the order (number of time lags) of the autoregressive model
# d is the degree of differencing (the number of times the data have had past values subtracted)
# q is the order of the moving-average model
# P, D, Q refer to the autoregressive, differencing, and moving average terms for the seasonal 
# components 

m = int(input.get("seasonality", 1))
arima = pm.AutoARIMA(trace=True, suppress_warnings=True, m=m)

pipeML = None

if "transformation" in input: 
  if input["transformation"] == "boxcox":
    pipeML = Pipeline([
       ("boxcox", BoxCoxEndogTransformer(lmbda2=1e-6)),
       ("arima",  arima)
    ])
  elif input["transformation"] == "log":
    pipeML = Pipeline([
       ("log", LogEndogTransformer(lmbda=1e-6)),
       ("arima",  arima)
    ])

if pipeML is None:
   pipeML = Pipeline([
       ("log", LogEndogTransformer(lmbda=1e-6)),
       ("arima",  arima)
   ])

pipeML.fit(y_train)

###########################################################################
# Report the metrics from the model run
###########################################################################

id = uuid.uuid1()

cdsw.track_metric("Model", str(pipeML.steps[-1][1].model_))
cdsw.track_metric("AIC", pipeML.steps[-1][1].model_.aic())
cdsw.track_metric("BIC", pipeML.steps[-1][1].model_.bic())
cdsw.track_metric("Id", id.hex)

####p#######################################################################
# Save the model
###########################################################################

modelFile = "airlines_pipeML_arima_{}.pk1".format(id.hex)
joblib.dump(pipeML, modelFile)

s3_client = boto3.client("s3")
response = s3_client.upload_file(modelFile, "kdavis-pse-demo", modelFile)
print(response)
