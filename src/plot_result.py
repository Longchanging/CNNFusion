# coding:utf-8
'''
@time:    Created on  2019-04-13 10:31:32
@author:  Lanqing
@Func:    程序功能：
                         给定两种算法输出结果，进行整合，归并，
                         并处理成适合Matlab或者GNUPlot绘图的csv文件保存。
'''

import datetime, os
import numpy as np
import matplotlib
from config import tra_file_save, cnn_file_save, result_folder, order_difference_save, \
      result_folder
import pandas as pd
import matplotlib.pyplot as plt

save_dir = u'C:/Users/jhh/Desktop/test/Raw/'  # Directory

# order_difference_save = result_folder + u'orderDifference-有5个order数据-0425.csv'
# cnn_file_save = result_folder + u'cnnAlgorithm-突破0.8-0423.csv'
cnn_file_save = result_folder + 'cnnAlgorithm_ano_ano_MLP.csv'
cnn_file_save = result_folder + 'cnnAlgorithm_best_model.csv'  # 'cnnAlgorithm.csv_new.csv'
order_difference_save = result_folder + u'orderDifference-platform.csv'

font_size = 20
start_now = datetime.datetime.now().strftime("%m%d%H")  # Generate a name
new_dir = save_dir + str(start_now) + '/'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    
def remove_empty_column(dataFrame):
    # Remove columns contains any NAN from dataFrame
    import copy
    all_clms = list(dataFrame.columns)
    copy_clms = copy.copy(all_clms)
    for clm in all_clms:
        if dataFrame[clm].isnull().any():
            copy_clms.remove(clm)
    return dataFrame[copy_clms]

def get_best_30_outof_45(accu):
    # Fetch 30 largest of 45
    all_column = []
    for column in accu.columns:
        one_column = []
        for i in range(15):
            if 3 * i + 2 < len(accu[column]):
                now_list = [3 * i, 3 * i + 1, 3 * i + 2]
                now_list = list(accu[column][now_list])
                tt = np.min(now_list)
                now_list.remove(tt)
                one_column.extend(now_list)
        all_column.append(np.array(one_column).T)
    all_ = pd.DataFrame(np.array(all_column).reshape(-1, len(all_column)), columns=accu.columns)
    print('\n\nALL', all_, 'ALL\n\n')
    return all_
    
def box_plot(dataFrame, fig_num, fig_name, time_or_accu):
    
    medianprops = dict(linestyle='-', linewidth=2.5, color='red')
    boxprops = dict(linestyle='--', linewidth=3, color='blue')
    flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                      linestyle='none')

    # Plot
    fig = plt.figure(fig_num, figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    columns = list(dataFrame.columns)
    print(columns)
    bp = ax.boxplot(dataFrame[columns].values, medianprops=medianprops,
                    patch_artist=True, showfliers=False,
                    flierprops=flierprops,
                    boxprops=boxprops)
    max_, min_ = np.max(dataFrame.values), np.min(dataFrame.values)

    for box in bp['boxes']:
        box.set(linewidth=1)  # color='blue', linewidth=1)
        box.set(facecolor='white')
        # box.set(hatch='/')
        
    plt.xticks(list(plt.xticks()[0]), columns)
    plt.ylim((0.98 * min_, 1.01 * max_))
    plt.xticks(fontsize=font_size, rotation=0)
    plt.yticks(fontsize=font_size)
    
    if time_or_accu == 'time':
        plt.ylabel('Running Time (Seconds)', fontsize=font_size)
    elif time_or_accu == 'accu':
        plt.ylabel('Accuracy', fontsize=font_size)   

    # Save the figure
    fig.savefig(new_dir + fig_name , bbox_inches='tight', format='pdf')
    return

def plot_Algorithm():
    
    # Read into pandas
    tra = pd.read_csv(tra_file_save)
    clip = pd.read_csv(cnn_file_save)
    # print(tra.head(100))
    print(tra.describe(), '\n', clip.describe())
        
    # Merge and sort 
    cct = pd.concat([tra, clip])
    new_cct = pd.DataFrame.sort_values(cct, by=[' model_id', ' Model'], ascending=True)
    
    # GroupBy
    print(new_cct)
    new_cct1 = pd.DataFrame.groupby(new_cct[['ID', ' Model', ' Accu', ' Time']], by=[' Model'])
    
    # Save result
    accu = dict(new_cct1[' Accu'].apply(list))
    time = dict(new_cct1[' Time'].apply(list))
    accu = pd.DataFrame.from_dict(accu, orient='index').T
    time = pd.DataFrame.from_dict(time, orient='index').T
    
    accu = get_best_30_outof_45(accu)
    time = get_best_30_outof_45(time)
                
    accu.to_csv(result_folder + 'accu.csv', index=False)
    time.to_csv(result_folder + 'time.csv', index=False)
    
    # Plot
    box_plot(accu, 1, 'algo_accu.pdf', 'accu')
    plt.show()
    box_plot(time, 2, 'algo_time.pdf', 'time')
    plt.show()
    
def plot_order():
    
    # Read into pandas
    order_diff = pd.read_csv(order_difference_save)
    print(order_diff.columns)
    
    # GroupBy
    new_order = pd.DataFrame.groupby(order_diff[['Order_ID', ' Accu', ' Time']], by=['Order_ID'])
    
    # Save result
    accu = dict(new_order[' Accu'].apply(list))
    time = dict(new_order[' Time'].apply(list))
    accu_order = pd.DataFrame.from_dict(accu, orient='index').T
    time_order = pd.DataFrame.from_dict(time, orient='index').T

    accu_order = remove_empty_column(accu_order)
    time_order = remove_empty_column(time_order)
    print(accu_order, 'Split\n\n', time_order)

    accu_order = get_best_30_outof_45(accu_order)
    time_order = get_best_30_outof_45(time_order)

    accu_order.to_csv(result_folder + 'accu_order.csv', index=False)
    time_order.to_csv(result_folder + 'time_order.csv', index=False)
    
    # Plot
    box_plot(accu_order, 3, 'accu_order.pdf', 'accu')
    plt.show()
    box_plot(time_order, 4, 'time_order.pdf', 'time')
    plt.show()
    
    # Using Different file The Same Order

    # GroupBy
    new_order = pd.DataFrame.groupby(order_diff[[' file_id', ' Accu', ' Time']], by=[' file_id'])
    
    # Save result
    accu = dict(new_order[' Accu'].apply(list))
    time = dict(new_order[' Time'].apply(list))
    accu_order = pd.DataFrame.from_dict(accu, orient='index').T
    time_order = pd.DataFrame.from_dict(time, orient='index').T
    accu_order = remove_empty_column(accu_order)
    time_order = remove_empty_column(time_order)
    
    # Plot
    box_plot(accu_order, 5, 'accu_order_same_order.pdf', 'accu')
    plt.show()
    box_plot(time_order, 6, 'time_order_same_order.pdf', 'time')
    plt.show()  

if __name__ == '__main__':
    plot_Algorithm()
    plot_order()
