
import pandas as pd
import numpy as np
import timeit
import os
import json
import joblib
import training
import ingestion

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = '/trainedmodel.pkl'
test_data_name = 'testdata.csv'
dataset_name = '/finaldata.csv'

# Function to get model predictions
def model_predictions():
    # read the deployed model and a test dataset, calculate predictions
    test_data = pd.read_csv('./{}/{}'.format(test_data_path, test_data_name))

    x_test = test_data.drop(['corporation', 'exited'], axis=1)

    model = joblib.load(prod_deployment_path + model_path)

    y_pred = model.predict(x_test)

    # return value should be a list containing all predictions
    return y_pred

# Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    data = pd.read_csv(dataset_csv_path + dataset_name)
    data = data.drop(['Unnamed: 0'], axis=1)

    summary = []
    lastmonth_activity = []
    lastyear_activity = []
    number_of_employees = []

    lastmonth_activity_mean = np.mean(data['lastmonth_activity'])
    lastmonth_activity_med = np.median(data['lastmonth_activity'])
    lastmonth_activity_std = np.std(data['lastmonth_activity'])

    lastmonth_activity.append(lastmonth_activity_mean)
    lastmonth_activity.append(lastmonth_activity_med)
    lastmonth_activity.append(lastmonth_activity_std)

    lastyear_activity_mean = np.mean(data['lastyear_activity'])
    lastyear_activity_med = np.median(data['lastyear_activity'])
    lastyear_activity_std = np.std(data['lastyear_activity'])

    lastyear_activity.append(lastyear_activity_mean)
    lastyear_activity.append(lastyear_activity_med)
    lastyear_activity.append(lastyear_activity_std)

    number_of_employees_mean = np.mean(data['number_of_employees'])
    number_of_employees_med = np.median(data['number_of_employees'])
    number_of_employees_std = np.std(data['number_of_employees'])

    number_of_employees.append(number_of_employees_mean)
    number_of_employees.append(number_of_employees_med)
    number_of_employees.append(number_of_employees_std)

    summary.append(lastmonth_activity)
    summary.append(lastyear_activity)
    summary.append(number_of_employees)

    # return value should be a list containing all summary statistics
    return summary

# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    exec_time = []
    files = [training.train_model(), ingestion.merge_multiple_dataframe()]

    for file in files:
        start = timeit.default_timer()
        file
        end = timeit.default_timer()
        time = (end-start)
        exec_time.append(time)

    # return a list of 2 timing values in seconds
    return exec_time

# Function to check dependencies
def outdated_packages_list():
    #get a list of
    outdate = os.system("pip list --outdated")
    return outdate


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
