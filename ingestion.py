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
final_data_path = 'finaldata.csv'
record_path = './ingesteddata/ingestedfiles.txt'


def write_records():
    date_time_object = datetime.now()
    now = str(date_time_object.year) + '/' + str(date_time_object.month) + '/' + str(date_time_object.day)
    data = pd.read_csv('./{}/{}'.format(output_folder_path, final_data_path))
    all_records = [output_folder_path, record_path, len(data.index), now]

    file = open(record_path, 'w')
    for element in all_records:
        file.write(str(element))
        file.write('\n')
    file.close()


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
        final_dataframe.to_csv('./{}/{}'.format(output_folder_path, final_data_path))
    else:
        final_dataframe.to_csv('./{}/{}'.format(output_folder_path, final_data_path))

    write_records()


if __name__ == '__main__':
    merge_multiple_dataframe()
