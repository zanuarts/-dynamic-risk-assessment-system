from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
data_path = '/finaldata.csv'
model_name = 'trainedmodel.pkl'


# Function for training the model
def train_model():
    data = pd.read_csv(os.getcwd() + '/' + dataset_csv_path + data_path)

    print(data.columns)

    x = data.drop(['Unnamed: 0', 'corporation', 'exited'], axis=1)
    y = data['exited']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    print(x_train, y_train)

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False
    )

    # fit the logistic regression to your data
    model.fit(x_train, y_train)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    if os.path.isdir(model_path) is False:
        os.makedirs(model_path)
        joblib.dump(model, './{}/{}'.format(model_path, model_name))
    else:
        joblib.dump(model, './{}/{}'.format(model_path, model_name))


if __name__ == '__main__':
    train_model()
