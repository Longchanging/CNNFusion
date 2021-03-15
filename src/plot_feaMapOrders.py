# coding:utf-8
'''
@time:    Created on  2019-05-10 09:53:05
@author:  Lanqing
@Func:    SEED.src.plot_feaMapOrders
'''

import pandas as pd, numpy as np, matplotlib.pyplot as plt

file = u'../images/不同order-featureMap结果展示（遍历）.txt'
fid = open(file, 'r')

accuracys = []
for line in fid:
    line = line.strip('\n')
    match = line.split('Further convolution Accuracy: ')[1:]
    if match:
        accuracy = float(match[0].split("%")[0])
        accuracys.append(accuracy)

x = range(len(accuracys))
y = accuracys
x = np.array(x).T
y = np.array(y).T
data = np.vstack((x, y)).T
print(data)
np.savetxt('../images/order_result.csv', data, delimiter=',', fmt='%.3f')

plt.plot(x, y)
plt.show()

plt.boxplot(y)
plt.show()
