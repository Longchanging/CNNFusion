# coding:utf-8
'''
@time:    Created on  2019-05-06 16:42:24
@author:  Lanqing
@Func:    尝试新方法。算法过程如下：
            1.全局采样,做自互相关计算
            2.找出分组
            3.定义变量， 定义 sensor based网络
            4.各自卷积得到 feature map
            5.尝不同feature map找 feature map order 
            6. feature map order有可能 class label关联
            7.用小而深model尝试map，用final model分类
            8.得到分类结果
'''

import sklearn
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from config import input_path, window_size, load_max_rows, epochs, batch_size, model_folder, tra_file_save, train_clips, validate_clips, test_clips, cnn_file_save, result_folder
from algorithms import vstack_list, final_model, gauss_filter, fft_transform, shuffle_train, PCA, n_component
from algorithms import naive_bayes_classifier, knn_classifier, decision_tree_classifier, rf_classifier, lr_classifier, svm_classifier, final_model_extractedFeatures_old
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import metrics 
import scipy.io as sio
import numpy as np
import time
import os
from keras.utils import to_categorical
import keras as K

use_person = 10
filter_num = 16
input_shape = (62, 5)
nClasses = 3

# The random seed
np.random.seed(10)

# Confusion
def confusion_plot(array):
    cf_figsize = (8, 6)
    font_size = 20
    cmap = plt.cm.Blues  # @UndefinedVariable
    save_dir = ''
    cm = np.array(array)
    target_names = list(range(62))
    normalize = False
    accuracy = np.trace(cm) / float(np.sum(cm))
    fig = plt.figure(figsize=cf_figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, range(1, len(target_names) + 1), rotation=True,
                    fontsize=font_size - 15)
        plt.yticks(tick_marks, target_names, fontsize=font_size - 15)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    res = ax.imshow(np.array(cm), cmap,
                    interpolation='nearest')
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    cb = plt.colorbar(res)  # , pad=0.01)
    plt.ylabel('Channels Correlations', fontsize=font_size + 5)
    plt.xlabel('Channels Correlations', fontsize=font_size + 5)
    plt.savefig(save_dir + 'aa.eps', format='eps')    
    plt.savefig(save_dir + 'aa.pdf', format='pdf')
    plt.show()
    return


# Attention Please!
# Change this file will affect all other files, 
# Especially the preprocess procedure!

def read_a_mat(eg_data, label):
    '''
        given clips eeg data of one person, preprocess and merge them 
    '''
    # Final Data to return
    train_data, train_label, test_data, test_label, \
        validate_data, validate_label = [], [], [], [] , [], []        
    
    # Loop All Clips of A man
    for i in range(15):
        all_sample_number, all_samples = [], []
        for key_word in ['de_LDS', 'de_movingAve']:
            A_key = key_word + str(i + 1)
            clip_a_data = eg_data[A_key]
            _, b, _ = clip_a_data.shape
            all_sample_number.append(b)
            
        # Min Sample
        min_sample = np.min(all_sample_number)
        for key_word in ['de_LDS', 'de_movingAve']:
            A_key = key_word + str(i + 1)
            clip_a_data = eg_data[A_key]
            clip_a_data = np.transpose(clip_a_data, (1, 0, 2))
            clip_a_data = clip_a_data[:min_sample, :, :]
            all_samples.append(clip_a_data)
            
        # A Clip Data
        clip_data = vstack_list(all_samples)
        clip_label = label[i] + 1 
        clip_label = clip_label * np.ones([int(len(clip_data)), 1])
        # print('\tClip %d shape:' % (i + 1), clip_data.shape)

            
        # Train and Test
        if (i + 1) in train_clips:
            train_data.extend(clip_data)
            train_label.extend(clip_label)
        elif (i + 1) in test_clips:
            test_data.extend(clip_data)
            test_label.extend(clip_label)   
        elif (i + 1) in validate_clips:
            validate_data.extend(clip_data)
            validate_label.extend(clip_label)     
    
    # Join data
    one_train_data = np.transpose(np.array(train_data), (0, 2, 1))
    one_train_label = np.array(train_label).reshape(-1, 1)
    one_test_data = np.transpose(np.array(test_data), (0, 2, 1))
    one_test_label = np.array(test_label).reshape(-1, 1)
    one_validate_data = np.transpose(np.array(validate_data), (0, 2, 1))
    one_validate_label = np.array(validate_label).reshape(-1, 1)
    
    # print(np.array(one_train_data).shape, np.array(one_test_data).shape, np.array(one_validate_data).shape)
    # print(one_train_label.shape, one_test_label.shape, one_validate_data.shape)
    
    return one_train_data, one_train_label, one_validate_data, \
           one_validate_label, one_test_data, one_test_label

           
def load_all_person(label):
    ''' 
        Load all person's eeg files
        Fetch all data for train, test, validate
    '''
    
    all_train_data, all_train_label, all_test_data, all_test_label, \
         all_validate_data, all_validate_label = [], [], [], [], [], []
    label = sio.loadmat(input_path + 'label.mat')['label'][0]  
    i = 0 
    
    for _, _, c in os.walk(input_path):
        for item in c:
            if '_' in item and i <= use_person:
                
                i += 1
                
                # Prepare data 
                tmp_file = sio.loadmat(input_path + item, verify_compressed_data_integrity=False)

                one_train_data, one_train_label, one_validate_data, one_validate_label, \
                       one_test_data, one_test_label = read_a_mat(tmp_file, label)


                # print('One person Shapes:', one_train_data.shape, one_validate_data.shape, \
                #       one_test_data.shape)
                all_train_data.append(one_train_data)
                all_train_label.append(one_train_label)
                all_validate_data.append(one_validate_data)
                all_validate_label.append(one_validate_label)
                all_test_data.append(one_test_data)
                all_test_label.append(one_test_label)
                
    all_train_data = vstack_list(all_train_data)    
    all_train_label = vstack_list(all_train_label)
    all_validate_data = vstack_list(all_validate_data)    
    all_validate_label = vstack_list(all_validate_label)
    all_test_data = vstack_list(all_test_data)    
    all_test_label = vstack_list(all_test_label)
    
    # Shuffle
    # all_train_data, all_train_label = shuffle_train(all_train_data, all_train_label)
    
    return all_train_data, all_train_label, all_validate_data, \
           all_validate_label, all_test_data, all_test_label
           
def get_corr_matrix(X_train):
    ''' 
        given raw data of many persons (X_train), get the correlation matrix
    '''
    corr_data = np.transpose(X_train, [2, 0, 1])
    # corr_data = corr_data()
    # corr_data = corr_data.reshape([corr_data.shape[0], corr_data.shape[1] * corr_data.shape[2]])
    print('corr_data shape:', corr_data.shape)
    corr_data = np.mean(corr_data, axis=2)
    # corr_data = corr_data[:, :, 1]
    print('corr_data shape:', corr_data.shape)
    corr_result = np.corrcoef(corr_data)
    np.savetxt('../images/corr.csv', corr_result, fmt='%.3f', delimiter=',')
    print(corr_result.shape, type(corr_result))
    return corr_result
    
def get_dict_groups(dbscan_list):
    '''
        given DBSCAN result list, find the dict groups 
    '''
    final_dict = {}
    total_groups = np.max(dbscan_list) - np.min(dbscan_list)
    for i in range(total_groups + 1):
        locs = list(np.where(dbscan_list == (np.min(dbscan_list) + i))[0])
        final_dict[i] = locs
    return final_dict

def get_groups():
    ''' 
        given correlation matrix, use DBSCAN to get groups 
    '''
    # corr_result = get_corr_matrix()
    # Get matrix
    corr_result = np.loadtxt('../images/corr.csv', delimiter=',')
    # Convert to distance
    corr_result = -corr_result + 1
    # confusion_plot(corr_result)
    # Cluster
    db = DBSCAN(eps=0.2, metric="precomputed", min_samples=2).fit(corr_result)
    db_result = list(db.labels_)
    print (db_result, len(db_result))
    final_groups = get_dict_groups(db_result)
    return final_groups
  
def sensorBasedEarlyFusion(final_groups):    
    
    from keras.models import Sequential
    from keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Merge, Dropout
    
    branchs = []
    for key in final_groups.keys():
        channels_to_use = final_groups[key] 
        channels_num = len(channels_to_use)
        tmp_branch = Sequential()
        tmp_branch.add(Reshape(target_shape=(window_size, channels_num, 1), input_shape=(window_size, channels_num)))
        tmp_branch.add(Conv2D(filter_num, (1, channels_num), activation='relu'))
        tmp_branch.summary()
        branchs.append(tmp_branch)
            
    merged = Merge(branchs, mode='concat')
    
    model = Sequential()
    model.add(merged)
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(nClasses, activation='softmax')) 
    # model.summary()
    return model

def sensorBased_UseOrder(final_groups, order):    
     
    ''' 
        given group orders, generate different models in "Merged" branches  
    '''
    
    from keras.models import Sequential
    from keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Merge, Dropout, Permute
    
    # Define Sensor based fusion
    branchs = []
    for key in order:  # Replace final_groups.keys()
        channels_to_use = final_groups[key] 
        channels_num = len(channels_to_use)
        tmp_branch = Sequential()
        tmp_branch.add(Reshape(target_shape=(window_size, channels_num, 1), input_shape=(window_size, channels_num)))
        tmp_branch.add(Conv2D(filter_num, (1, channels_num), activation='relu'))
        branchs.append(tmp_branch)
        
    # Merge
    # Use Order here
    merged = Merge(branchs, mode='concat')
    
    # Sensor based Model
    model = Sequential()
    model.add(merged)
    model.add(Permute((1, 3, 2)))  # Transpose the dimensions
    
    # We can use dropout and pooling here

    model.add(Conv2D(filter_num, (1, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    model.add(Conv2D(filter_num, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    model.add(Conv2D(filter_num * 2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(nClasses, activation='softmax')) 
    model.summary()
    return model


def final_model_extractedFeatures_MLP():
    from keras.layers.core import Dense
    from keras.models import Sequential
    from keras.layers import  Reshape, Flatten, Conv2D
    from keras.layers import Dropout
    from keras.layers import MaxPooling2D
    # Define Model
    model = Sequential()
    model.add(Reshape(target_shape=(window_size, 62, 1), input_shape=(window_size, 62)))
    model.add(Flatten())
    # 3 layer MLP
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(3, activation='softmax')) 
    model.summary()
    return model

def printModel(final_groups, order):    
     
    ''' 
        given group orders, generate different models in "Merged" branches  
    '''
    
    from keras.models import Sequential
    from keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Merge, Dropout, Permute
    
    # Define Sensor based fusion
    branchs = []
    for key in order:  # Replace final_groups.keys()
        channels_to_use = order
        channels_num = len(key)
        tmp_branch = Sequential()
        tmp_branch.add(Reshape(target_shape=(window_size, channels_num, 1), input_shape=(window_size, channels_num)))
        tmp_branch.add(MaxPooling2D(pool_size=(1, 2), strides=1, padding='same'))
        tmp_branch.add(Conv2D(filter_num, (1, channels_num), activation='relu'))
        branchs.append(tmp_branch)
        
    # Merge
    # Use Order here
    merged = Merge(branchs, mode='concat')
    
    # Sensor based Model
    model = Sequential()
    model.add(merged)
    model.add(Permute((1, 3, 2)))  # Transpose the dimensions
    
    # We can use dropout and pooling here

    model.add(Conv2D(filter_num, (1, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    model.add(Conv2D(filter_num, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    model.add(Conv2D(filter_num * 2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(nClasses, activation='softmax')) 
    model.summary()
    return model

def goodModel():
    
    from keras.models import Sequential
    from keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Merge, Dropout

    model = Sequential()
    model.add(Reshape(target_shape=(window_size, 62, 1), input_shape=(window_size, 62)))
    model.add(Conv2D(filter_num, (1, 9), activation='relu'))
    model.add(Conv2D(filter_num, (1, 9), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(filter_num, (1, 7), activation='relu'))
    model.add(Conv2D(filter_num, (1, 7), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(filter_num, (1, 7), activation='relu'))
    model.add(Conv2D(filter_num, (1, 5), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax')) 
    model.summary()

    return model


def shuffle__data(total_groups, iterations):
    ''' 
        Generate the orders
    '''
    import random
    orders = []
    a = list(np.arange(0, total_groups, 1))
    orders.append(a)
    for _ in range(iterations):
        idx = random.sample(a, total_groups)
        orders.append(idx)
    return orders

if __name__ == '__main__':

#     #-------------           Step 0: Prepare data            ------------------------#
#   
#     label = sio.loadmat(input_path + 'label.mat')['label'][0]  
#     X_train, y_train_org , X_validate, y_validate, X_test, y_test = load_all_person(label)
#     # One Hot
#     y_train = to_categorical(y_train_org)
#     y_test = to_categorical(y_test)
#     y_validate = to_categorical(y_validate)
#     print('Final data shape: \t', X_train.shape, y_train.shape , X_validate.shape, y_validate.shape, X_test.shape, y_test.shape)
# 
#     #-------------          Step 1: Get the groups          ------------------------#
#     
#     final_groups__all = []
# 
#     corr_result = get_corr_matrix(X_train)
#     final_groups = get_groups()
#     final_groups__all.append(final_groups)
#     print(final_groups, '\n\n\n')
#     
#     #------- Step 1-1 : Get the groups  and compare the class difference -----------#
#     max_, min_ = np.max(y_train_org), np.min(y_train_org)
#     for i in range(int(min_), int(max_) + 1):
#         locs = np.where(y_train_org == i)[0]
#         print(i)
#         X_train_one_class = X_train[locs]
#         corr_result = get_corr_matrix(X_train_one_class)
#         final_groups = get_groups()
#         final_groups__all.append(final_groups)
#         print('Class %d' % i, '\n', final_groups)
#         
#     big_groups = {}
#     i = 0
#     for final_groups in final_groups__all:
#         for key in final_groups.keys():
#             big_groups[i] = final_groups[key]
#             i += 1
#     print(big_groups)
#     
#     final_groups = big_groups
# 
#     #-------------      Step 2: Use Sensor Based Method     ------------------------#
#      
#     # prepare sensor based data
#     #         X_train_list, X_test_list, X_validate_list = [], [], []
#     #         for key in final_groups.keys():  # 0,1,2,3,4, increase
#     #             X_train_list.append(X_train[:, :, final_groups[key]])
#     #             X_validate_list.append(X_validate[:, :, final_groups[key]])
#     #             X_test_list.append(X_test[:, :, final_groups[key]])
#          
#     #     #-------------  Step 3: Train normal model as baseline  ------------------------#
#     #     
#     #     unique_name = '%.3f' % (time.time() % 100000)
#     #     callbacks_list = [
#     #         K.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10),  # 5 epoch no improvement
#     #         K.callbacks.ModelCheckpoint(model_folder + 'MyModel_best_%s.5df5' % unique_name,
#     #                     monitor='val_loss', verbose=2, save_best_only=True,  # default,val_loss
#     #                     mode='min', period=1)
#     #                       ] 
#     #     # MLP
#     #     model = final_model_extractedFeatures_MLP()
#     #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
#     #     model.fit(X_train, y_train, epochs=epochs + 100, validation_data=(X_validate, y_validate),
#     #               callbacks=callbacks_list,
#     #               verbose=2, batch_size=batch_size)
#     #     scores = model.evaluate(x=X_test, y=y_test, verbose=0)
#     #     end_time3 = time.time()  # Show Info
#     #     print("MLP Accuracy: %.2f%%" % (scores[1] * 100))
#     #     
#     #     # Good Model
#     #     model = goodModel()
#     #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
#     #     model.fit(X_train, y_train, epochs=epochs + 100, validation_data=(X_validate, y_validate),
#     #               callbacks=callbacks_list,
#     #               verbose=2, batch_size=batch_size)
#     #     scores = model.evaluate(x=X_test, y=y_test, verbose=0)
#     #     end_time3 = time.time()  # Show Info
#     #     print("MLP Accuracy: %.2f%%" % (scores[1] * 100))
#     #       
#     #     #-------------   Step 4: Train Our sensor based model  ------------------------#
#     #      
#     #     unique_name = '%.3f' % (time.time() % 100000)
#     #     callbacks_list = [
#     #         K.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10),  # 5 epoch no improvement
#     #         K.callbacks.ModelCheckpoint(model_folder + 'MyModel_best_%s.5df5' % unique_name,
#     #                     monitor='val_loss', verbose=2, save_best_only=True,  # default,val_loss
#     #                     mode='min', period=1)
#     #                       ] 
#     #     # Sensor based
#     #     model = sensorBasedEarlyFusion(final_groups)
#     #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
#     #     model.fit(X_train_list, y_train, epochs=epochs + 100, validation_data=(X_validate_list, y_validate),
#     #               callbacks=callbacks_list,
#     #               verbose=2, batch_size=batch_size)
#     #     scores = model.evaluate(x=X_test_list, y=y_test, verbose=0)
#     #     end_time3 = time.time()  # Show Info
#     #     print("Sensor Based Accuracy: %.2f%%" % (scores[1] * 100))
#      
#     #-------------      Step 5: Test Further convolution        --------------------#
#  
#     orders = shuffle__data(len(final_groups.keys()), 300)
#     # orders = [[0, 1, 2, 3, 4, 5], [2, 4, 5, 1, 3, 0], [2, 4, 5, 3, 0, 1]] orders = [ [0, 1, 2, 3, 4, 5], [2, 4, 5, 1, 3, 0], [5, 2, 3, 1, 0, 4] ]
#  
#     count_iterations = 0
#          
#     for order in orders:
#          
#         count_iterations += 1
#         print('The %d iterations:' % count_iterations, '\nGroup used:', final_groups, '\n The order:\n', order)
#          
#         # prepare sensor based data
#         X_train_list, X_test_list, X_validate_list = [], [], []
#         for key in order:  # 0,1,2,3,4, increase
#             X_train_list.append(X_train[:, :, final_groups[key]])
#             X_validate_list.append(X_validate[:, :, final_groups[key]])
#             X_test_list.append(X_test[:, :, final_groups[key]])
#  
#         model = sensorBased_UseOrder(final_groups, order)
#         unique_name = '%.3f' % (time.time() % 100000)
#         callbacks_list = [
#             K.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10),  # 5 epoch no improvement
#             K.callbacks.ModelCheckpoint(model_folder + 'MyModel_best_%s.5df5' % unique_name,
#                         monitor='val_loss', verbose=0, save_best_only=True,  # default,val_loss
#                         mode='min', period=2)
#                           ] 
#         # Test Further convolution
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
#         model.fit(X_train_list, y_train, epochs=epochs + 100, validation_data=(X_validate_list, y_validate),
#                   callbacks=callbacks_list,
#                   verbose=0, batch_size=batch_size)
#         scores = model.evaluate(x=X_test_list, y=y_test, verbose=0)
#         end_time3 = time.time()  # Show Info
#         print("Further convolution Accuracy: %.2f%%" % (scores[1] * 100))
    orders = [[0, 1, 2, 3, 4, 5], [2, 4, 5, 1, 3, 0], [2, 4, 5, 3, 0, 1]] 
    orders = [ [0, 1, 2, 3, 4, 5], [2, 4, 5, 1, 3, 0], [2, 4, 1, 3, 0], [2, 4, 5, 1, 32, 3, 4, 4, 5, 2, 4, 5, 1, 32, 3, 4, 4, 5, ], [2, 4, 5, 1, 32, 3, 4, 4, 5, 56, 6, 6, 7, 7, 5, 0], [5, 2, 3, 1, 0, 4] ]
    finalGroups = range(100)
    # printModel(finalGroups, orders)
    goodModel()
