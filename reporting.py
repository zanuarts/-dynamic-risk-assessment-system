import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib



# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
test_data_name = 'testdata.csv'
model_path = os.path.join(config['output_model_path'])
model_name = 'trainedmodel.pkl'


# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    test_data = pd.read_csv('./{}/{}'.format(test_data_path, test_data_name))

    x_test = test_data.drop(['corporation', 'exited'], axis=1)
    y_test = test_data['exited']

    model = joblib.load('./{}/{}'.format(model_path, model_name))
    y_pred = model.predict(x_test)
    result = metrics.confusion_matrix(y_test, y_pred)
    # write the confusion matrix to the workspace






if __name__ == '__main__':
    score_model()
