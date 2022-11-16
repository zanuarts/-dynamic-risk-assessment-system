import shutil

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



# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])
model_name = '/trainedmodel.pkl'
last_score_path = 'lastestscore.txt'
ingested_file_path = './ingesteddata/ingestedfiles.txt'

print(model_path+model_name)

# function for deployment
def store_model_into_pickle():
    # copy the latest pickle file,
    # the latestscore.txt value,
    # and the ingestfiles.txt file into the deployment directory
    if os.path.isdir(prod_deployment_path) is False:
        os.makedirs(prod_deployment_path)
        shutil.copy(model_path+model_name, prod_deployment_path)
        shutil.copy(last_score_path, prod_deployment_path)
        shutil.copy(ingested_file_path, prod_deployment_path)
    else:
        shutil.copy(model_path + model_name, prod_deployment_path)
        shutil.copy(last_score_path, prod_deployment_path)
        shutil.copy(ingested_file_path, prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle()
        
        
        

