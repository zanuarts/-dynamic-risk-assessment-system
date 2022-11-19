from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics
import scoring
import json
import os



# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    # call the prediction function you created in Step 3
    result = diagnostics.model_predictions()
    # add return value for prediction outputs
    return result

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    # check the score of the deployed model
    score = scoring.score_model()
    # add return value (a single F1 score number)
    return score

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    # check means, medians, and modes for each column
    data_summary = diagnostics.dataframe_summary()
    # return a list of all calculated summary statistics
    return data_summary


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    # check timing and percent NA values
    timing = diagnostics.execution_time()
    # add return value for all diagnostics
    return timing

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
