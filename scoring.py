from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import joblib



# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
test_data_name = 'testdata.csv'
model_path = os.path.join(config['output_model_path'])
model_name = 'trainedmodel.pkl'
f1_score_path = 'lastestscore.txt'

# Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    test_data = pd.read_csv('./{}/{}'.format(test_data_path, test_data_name))

    x_test = test_data.drop(['corporation', 'exited'], axis=1)
    y_test = test_data['exited']

    model = joblib.load('./{}/{}'.format(model_path, model_name))
    y_pred = model.predict(x_test)
    result = metrics.f1_score(y_test, y_pred)

    f = open(f1_score_path, 'w')
    f.write('\nModel Performance on Test Data')
    f.write('\n-----')
    f.write('\nF1 Score: ' + str(result))
    f.write('\n-----\n')
    f.close()


if __name__ == '__main__':
    score_model()
