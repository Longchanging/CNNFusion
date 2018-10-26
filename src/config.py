# coding:utf-8
'''
@time:    Created on  2018-10-26 09:32:49
@author:  Lanqing
@Func:    Configuration of the whole project
'''

#### Read data and rolling window
input_path_base = 'E:/DATA/CNNFusion/'
tmp_path_base = 'E:/DATA/CNNFusion/tmp/'
activities = ['walking', 'standing', 'sitting', 'running', 'lying', 'jumping', 'climbingup', 'climbingdown']
sensors = ['Microphone', 'MagneticField', 'Light', 'Gyroscope', 'GPS', 'acc']
positions = ['waist', 'shin', 'head', 'upperarm', 'thigh', 'chest', 'forearm']
filterLowSampleRateData = 5000
rolling_mean_interval = 100  # millisecond 

#### Process and prepare data
process_pid_list = [1, 2, 5, 6, 15]
deep_learning_window_size, overlap_window = 20, 5
