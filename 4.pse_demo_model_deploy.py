###########################################################################
# @Team:     Cloudera Partner Solutions Engineering
# @Author:   kdavis@cloudera.com
# @Purpose:  Load the Auto ARIMA model and calculate predictions
# @Modified: 11152021 init
###########################################################################

import joblib
import json
from boto.s3.connection import S3Connection

def predictUsingArima(args):
  n = int(args["predictions"])
  s3Bucket = args["s3-bucket"]
  modelName = args["modelName"]
  
  aws_connection = S3Connection()
  bucket = aws_connection.get_bucket(s3Bucket)
  key = bucket.get_key(modelName)
  key.get_contents_to_filename(modelName)

  joblib_preds = joblib.load(modelName).predict(n_periods=n)

  return json.dumps(joblib_preds.tolist())

