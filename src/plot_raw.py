# coding:utf-8
'''
@time:    Created on  2019-04-16 22:24:07
@author:  Lanqing
@Func:    Using T-SNE
'''

from config import input_path
from deepCNN import load_all_person
from algorithms import shuffle_train
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import scipy.io as sio
import time, datetime
import os

save_dir = u'C:/Users/jhh/Desktop/test/Raw/'  # Directory

font_size = 22
start_now = datetime.datetime.now().strftime("%m%d%H")  # Generate a name
new_dir = save_dir + 'TSNE_' + str(start_now) + '/'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

if __name__ == '__main__':
    
    # The random seed
    np.random.seed(10)

    # Loop On all Experiments
    i = 0
    label = sio.loadmat(input_path + 'label.mat')['label'][0]  
    for _, _, c in os.walk(input_path):
        for item in c:
            if '_' in item :  # and i <= use_person
                i += 1
                j = 1
                result_dict = []   
                             
                # Prepare data
                start_time = time.time()
                tmp_file = sio.loadmat(input_path + item)
                # Train test split
                X_train, y_train , X_validate, y_validate, X_test, y_test = load_all_person(label, tmp_file)
                
                # 3 dimension reshape to 2 dimension (sample x features)
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
                X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1] * X_validate.shape[2])
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

                # Using T-SNE
                # Random select points from train and test set
                X , y = shuffle_train(X_train, y_train)
                X , y = X[:2000], y[:2000]
                X_t , y_t = shuffle_train(X_test, y_test)
                X_t , y_t = X_t[:2000], y_t[:2000]            
                
                tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
                X_tsne = tsne.fit_transform(X)
                X_t_tsne = tsne.fit_transform(X_t)
                
                print("Org data dimension is {}. \
                      Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
                
                '''  嵌入空间可视化    '''
                
                # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
                # x_t_min, x_t_max = X_t_tsne.min(0), X_t_tsne.max(0)
                # X_t_norm = (X_t_tsne - x_t_min) / (x_t_max - x_t_min)  # 归一化
                
                # Rewrite for another min-max
                x_max = np.max(np.concatenate([X_tsne, X_t_tsne]), axis=0)
                x_min = np.min(np.concatenate([X_tsne, X_t_tsne]), axis=0)
                X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
                X_t_norm = (X_t_tsne - x_min) / (x_max - x_min)  # 归一化
                
                '''   额外添加分类模块    '''
                from algorithms import  decision_tree_classifier, rf_classifier
                from sklearn import metrics 
                conventional_models = { 'DT':decision_tree_classifier}
                # Using All models
                for one_model in conventional_models.keys():
                    print('Now processing the %dth file, using the %dth model' % (i, j))
                    model = conventional_models[one_model]()
                    model = model.fit(X_norm, y)
                    y_predict = model.predict(X_t_norm)
                    # Predict
                    acc = metrics.accuracy_score(y_t, y_predict)
                    cm = metrics.confusion_matrix(y_t, y_predict)
                    end_time = time.time()
                    print(acc, '\n', cm)
                                
#                 plt.figure(figsize=(10, 10))
#                 for i in range(X_norm.shape[0]):
#                     plt.text(X_norm[i, 0], X_norm[i, 1], str(int(y[i][0])),
#                              color="red",
#                              fontdict={'weight': 'bold', 'size': 9})
#                     plt.text(X_t_norm[i, 0], X_t_norm[i, 1], str(int(y_t[i][0])),
#                              color="green",
#                              fontdict={'weight': 'bold', 'size': 9})
#                 plt.xticks([])
#                 plt.yticks([])
#                 plt.savefig(new_dir + '%s.pdf' % item, format='pdf')
#                 # plt.show()
