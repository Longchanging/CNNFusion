# coding:utf-8
'''
@time:    Created on  2018-10-26 09:35:31
@author:  Lanqing
@Func:    Process and prepare data
'''

import pandas as pd, numpy as np, os
from config import tmp_path_base, activities, process_pid_list, deep_learning_window_size, overlap_window

def extra_effort(one_dataframe):
    
    #### do extra operation to ensure the result is correct
    df_all = one_dataframe.sort_values(by='attr_time', ascending=True)  # sort
    tt = list(df_all.columns)
    for item in tt:
        if ('attr_time' in item) or ('Unnamed' in item):  # Attention on multiple conditions!!!!
            tt.remove(item)
    df_all = df_all[tt]
    df_all = df_all.values  # every column is a sensor 
        
    return df_all


def moving_window_one_df(array_position, activity):
    
    #### moving window and get output
    list_data, list_label, i = [], [], 0
    rows = len(array_position)
    while(i * overlap_window + deep_learning_window_size < rows):  # attention here
        tmp_window = array_position[(i * overlap_window) : (i * overlap_window + deep_learning_window_size), :]  
        i += 1
        list_data.append(tmp_window)
        list_label.append(activity)

    return list_data, list_label


def get_data():
    
    #### concat all data into 3-D array and record corresponding labels
    list_all_data, list_all_label = [], []
    for i in process_pid_list:  
        tmp_path = tmp_path_base + '/proband' + str(i) + '/'
        for activity in activities:  
            if os.path.exists(tmp_path + str(i) + '_' + activity + '.csv'):
                df_one_position = pd.read_csv(tmp_path + str(i) + '_' + activity + '.csv')  
                array_one_position = extra_effort(df_one_position)
                list_a_data, list_a_label = moving_window_one_df(array_one_position, activity)
                print('raw data shape: ', array_one_position.shape, 'after moving window shape: (%d,%d,%d)'
                      % (len(list_a_data), list_a_data[0].shape[0], list_a_data[0].shape[1]))
                if list_a_data[0].shape[1] == 70:  # all 27 sensors available
                    list_all_data.extend(list_a_data)
                    list_all_label.extend(list_a_label)
    
    new_array = np.array(list_all_data)
    new_array = new_array.reshape(-1, list_all_data[0].shape[0], list_all_data[0].shape[1])
    print('final data shape:', new_array.shape, 'final label shape:', len(list_all_label))
    
    np.save(tmp_path_base + 'data.npy', new_array)
    np.save(tmp_path_base + 'label.npy', np.array(list_all_label))
    
get_data()
