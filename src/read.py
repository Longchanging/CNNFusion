# coding:utf-8
'''
@time:    Created on  2018-10-24 19:08:31
@author:  Lanqing
@Func:    Read the HAR files of single man
'''

import os, pandas as pd, numpy as np, time

input_path_base = 'E:/DATA/CNNFusion/'
tmp_path_base = 'E:/DATA/CNNFusion/tmp/'
activities = ['walking', 'standing', 'sitting', 'running', 'lying', 'jumping', 'climbingup', 'climbingdown']
sensors = ['Microphone', 'MagneticField', 'Light', 'Gyroscope', 'GPS', 'acc']
positions = ['waist', 'shin', 'head', 'upperarm', 'thigh', 'chest', 'forearm']
filterLowSampleRateData = 5000
rolling_mean_interval = 100  # millisecond 

def Read_get_an_activity_MinMax(input_path_one_person, activity):

    #### find min_max in an activity of single person
    df_dict = {}
    has__sensors = False
    one_activity_min_max, one_activity_max, one_activity_min = [], 0, 0
    for (root, _, files) in os.walk(input_path_one_person):  # List all file names
        for filename in files:  
            if ('.csv' in filename) and (activity in filename): 
                per_file = os.path.join(root, filename)
                parent_name = root.split('/')[-3] 
                per_file_name = parent_name + '_' + filename 
                df = pd.read_csv(per_file) 
                if len(df) > filterLowSampleRateData:  # filter
                    df['attr_time'] = df['attr_time'].apply(lambda x : x % 10000000)  # Clock synchronization
                    one_activity_min_max.extend([np.max(df['attr_time']), np.min(df['attr_time'])])
                    name_list = []
                    for name_ in df.columns:  # rename
                        if name_ != 'attr_time':
                            name_ = filename.split('.')[0] + '_' + name_
                        name_list.append(name_)    
                    df.columns = name_list
                    tt = list(df.columns)  # remove id column
                    for item in tt:
                        if 'id' in item:
                            tt.remove(item)
                    df = df[tt]
                    df_dict[per_file_name] = df
                    
    if len(df_dict.keys()) > 1:
        has__sensors = True
        one_activity_max, one_activity_min = np.max(one_activity_min_max), np.min(one_activity_min_max)
    
    print('sensor Num:', len(df_dict.keys()))
        
    return df_dict, one_activity_max, one_activity_min, has__sensors
    

def merge_an_activity_data(activity, person_id, tmp_path, df_dict, one_activity_max, one_activity_min):
    
    #### merge one activity data
    df_all = pd.DataFrame() 
    
    for df_name in df_dict.keys():
        df = df_dict[df_name]
        max_interval, min_interval = int(one_activity_max / rolling_mean_interval) + 1, int(one_activity_min / rolling_mean_interval) - 1
        
        info = []
        for t in range(min_interval, max_interval):  # Fetch data of per time-stamp
            df_a = df[(df['attr_time'] < t * rolling_mean_interval) & (df['attr_time'] >= (t - 1) * rolling_mean_interval)]
            if not df_a.empty:
                df_now = df_a.mean(axis=0)
                df_now['attr_time'] = t * rolling_mean_interval
                info.append(df_now)
            
        new_df = pd.concat(info, axis=1).T
        print('Before rolling mean shape：', df.shape, 'After rolling mean shape：', new_df.shape)  # , new_df.iloc[1]) 
        
        if not df_all.empty:  # merge
            df_all = df_all.merge(new_df, on='attr_time', how='outer')
        else:
            df_all = new_df
            
    df_all = df_all.sort_values(by='attr_time', ascending=True)  # sort
    df_all = df_all.fillna(method='ffill', axis=0)  # file missing values
    
    print('All sensor merged shape:', df_all.shape)
    df_all.to_csv(tmp_path + str(person_id) + '_' + activity + '.csv')  # rename and save

def process_all_files_all_man():

    start_time = time.time()
    
    #### use functions to process all files
    for i in range(15):  # loop all users
        
        input_path = input_path_base + '/proband' + str(i + 1) + '/data/'
        tmp_path = tmp_path_base + '/proband' + str(i + 1) + '/'
        
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)  # make directory
            
        for activity in activities:  # loop all activities, avoid no sensors available
            df_dict, one_activity_max, one_activity_min, has__sensors = Read_get_an_activity_MinMax(input_path, activity)
            if has__sensors:
                merge_an_activity_data(activity, i + 1, tmp_path, df_dict, one_activity_max, one_activity_min)
            
                now_time = time.time()
                spend_time = now_time - start_time
                print('\nTill now time spent: %.2f seconds\n' % spend_time)
        
    return

process_all_files_all_man()
