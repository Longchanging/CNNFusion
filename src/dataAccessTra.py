# coding:utf-8
'''
@time:    Created on  2019-04-18 10:59:43
@author:  Lanqing
@Func:    SEED.src.dataAccess
'''

from config import input_path, window_size, load_max_rows, \
        epochs, batch_size, model_folder, tra_file_save, \
        train_clips, validate_clips, test_clips, cnn_file_save
from algorithms import vstack_list, final_model, \
        gauss_filter, fft_transform, shuffle_train, PCA, n_component
from algorithms import naive_bayes_classifier, knn_classifier, \
     decision_tree_classifier, rf_classifier, lr_classifier, svm_classifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import metrics 
import scipy.io as sio
import numpy as np
import time
import os

# The random seed
np.random.seed(10)
input_path = 'E:/DATA/SEED/seed_data/ExtractedFeatures/ExtractedFeatures/'

# Attention Please!
# Change this file will affect all other files, 
# Especially the preprocess procedure!

def read_a_mat(eg_data, label):
    
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
        print('\tClip %d shape:' % (i + 1), clip_data.shape)

            
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
    
    print(np.array(one_train_data).shape, np.array(one_test_data).shape, np.array(one_validate_data).shape)
    print(one_train_label.shape, one_test_label.shape, one_validate_data.shape)
    
    return one_train_data, one_train_label, one_validate_data, \
           one_validate_label, one_test_data, one_test_label

           
def load_all_person(label, tmp_file):
    ''' 
        Load all person's eeg files
        Fetch all data for train, test, validate
    '''
    all_train_data, all_train_label, all_test_data, all_test_label, \
         all_validate_data, all_validate_label = [], [], [], [], [], []
    one_train_data, one_train_label, one_validate_data, one_validate_label, \
           one_test_data, one_test_label = read_a_mat(tmp_file, label)
    print('One person Shapes:', one_train_data.shape, one_validate_data.shape, \
           one_test_data.shape)
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
    all_train_data, all_train_label = shuffle_train(all_train_data, all_train_label)
    
    print('Final data shape', all_train_data.shape, all_test_data.shape, all_validate_data.shape)
    return all_train_data, all_train_label, all_validate_data, \
           all_validate_label, all_test_data, all_test_label

if __name__ == '__main__':
    
    # Loop All models
    conventional_models = {
                           'NB':naive_bayes_classifier,
                           'KNN':knn_classifier,
                           'DT':decision_tree_classifier,
                           'RF':rf_classifier,
                           'LR':lr_classifier,
                           'SVM':svm_classifier,
                           }

    # Check result file
    if os.path.exists(tra_file_save):
        os.remove(tra_file_save)
        fid = open(tra_file_save, 'w')
        fid.write("ID, model_id, Model, Accu, Time")
        fid.write('\n')
        fid.close()

    # Loop On all Experiments
    i = 0
    label = sio.loadmat(input_path + 'label.mat')['label'][0]  
    for _, _, c in os.walk(input_path):
        for item in c:
            if '_' in item :  # and i <= use_person
                i += 1
                j = 8
                result_dict = []   
                
                # Prepare data 
                start_time = time.time()
                tmp_file = sio.loadmat(input_path + item, verify_compressed_data_integrity=False)
                print(item)
                # Train test split
                X_train, y_train , X_validate, y_validate, X_test, y_test = load_all_person(label, tmp_file)
                # 3 dimension reshape to 2 dimension (sample x features)
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
                X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1] * X_validate.shape[2])
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
                    
                # Using All models
                for one_model in conventional_models.keys():
                    print('Now processing the %dth file, using the %dth model' % (i, j))
                    model = conventional_models[one_model]()
                    model = model.fit(X_train, y_train)
                    y_predict = model.predict(X_test)
                    # Predict
                    acc = metrics.accuracy_score(y_test, y_predict)
                    cm = metrics.confusion_matrix(y_test, y_predict)
                    end_time = time.time()
                    result_dict.append([i, j, one_model, acc, '%.3f' % (end_time - start_time)])
                    j += 1
                    
                # Save result
                fid = open(tra_file_save, 'a')
                df = pd.DataFrame(result_dict, columns=['ID', 'model_id', 'Model', 'Accu', 'Time'])
                df.to_csv(fid, index=False, header=False)
                fid.close()
