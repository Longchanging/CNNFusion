# coding:utf-8
'''
@time:    Created on  2018-10-26 13:37:35
@author:  Lanqing
@Func:    Set baseline for the learning process before using CNN
'''

from config import tmp_path_base
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np, pandas as pd

new_array = np.load(tmp_path_base + 'data.npy',)
list_all_label = np.load(tmp_path_base + 'label.npy')
print(new_array.shape)

idx = np.random.randint(new_array.shape[0], size=40000)
print(idx)
new_array = new_array[idx]
list_all_label = list_all_label[idx]

data = new_array.reshape([new_array.shape[0], new_array.shape[1] * new_array.shape[2]])
le = LabelEncoder()
label = le.fit_transform(list_all_label)
df = pd.DataFrame(data)
df = df.fillna(method='ffill')
data = df.values

data = MinMaxScaler().fit_transform(data)

model2 = RandomForestClassifier(n_estimators=100)
scores2 = cross_val_score(model2, data, label, cv=10, scoring='accuracy', verbose=1)
print(scores2)
