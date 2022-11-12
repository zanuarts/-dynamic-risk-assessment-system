import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
final_data_file = 'finaldata.csv'


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    directories = ['/{}/'.format(input_folder_path)]
    final_dataframe = pd.DataFrame(
        columns=['corporation', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited'])

    for directory in directories:
        filenames = os.listdir(os.getcwd() + directory)
        for each_filename in filenames:
            try:
                current_df = pd.read_csv(os.getcwd() + directory + each_filename)
                final_dataframe = final_dataframe.append(current_df).reset_index(drop=True)
            except Exception as e:
                print(e)

    if os.path.isdir(output_folder_path) is False:
        print('hello')
        os.makedirs(output_folder_path)
        final_dataframe.to_csv('./{}/{}'.format(output_folder_path, final_data_file))
    else:
        final_dataframe.to_csv('./{}/{}'.format(output_folder_path, final_data_file))


if __name__ == '__main__':
    merge_multiple_dataframe()
