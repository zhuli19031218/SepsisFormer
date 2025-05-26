import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np



def sample_adaptation(source, target, source_url, target_url, adaptation_method='mean_teacher'):
    # 读取数据
    source_labels = source.pop("dead").to_numpy()
    source_data = source
    target_labels = target.pop("dead").to_numpy()
    target_data = target
    
    # 计算源域和目标域样本的均值和方差
    source_mean = np.mean(source_data, axis=0)
    source_var = np.var(source_data, axis=0)
    target_mean = np.mean(target_data, axis=0)
    target_var = np.var(target_data, axis=0)

    # 根据不同的适应方法进行重采样
    if adaptation_method == 'mean_teacher':
        # 使用均值教师模型的方法进行重采样
        source_data_adapted = (source_data - source_mean) * np.sqrt(target_var / source_var) + target_mean
        source_labels_adapted = source_labels
        target_data_adapted = (target_data - target_mean) * np.sqrt(source_var / target_var) + source_mean
        target_labels_adapted = target_labels
    elif adaptation_method == 'whitening':
        # 使用白化方法进行重采样
        source_data_adapted = (source_data - source_mean) / np.sqrt(source_var + 1e-6) * np.sqrt(
            target_var / source_var) + target_mean
        source_labels_adapted = source_labels
        target_data_adapted = (target_data - target_mean) / np.sqrt(target_var + 1e-6) * np.sqrt(
            source_var / target_var) + source_mean
        target_labels_adapted = target_labels
    else:
        raise ValueError("Invalid adaptation method.")

    source_labels = pd.DataFrame(source_labels)
    source_data_adapted=pd.concat([source_data_adapted,source_labels],axis=1)
    source_data_adapted.to_csv(source_url, index=False)
    target_labels = pd.DataFrame(target_labels)
    target_data_adapted=pd.concat([target_data_adapted,target_labels],axis=1)
    target_data_adapted.to_csv(target_url, index=False)

    return source_data_adapted, source_labels_adapted, target_data_adapted, target_labels_adapted


if __name__ == '__main__':
    # 方法
    adaptation_method='whitening'
    # 源域
    source = pd.read_csv(r"data\mimic3_eICU_smote_8.csv")
    # 目标域
    target = pd.read_csv(r"data\local_smote_8.csv")
    # 输出保存路径
    source_url="data/domain_adaptation_data/mimic3_eICU_smote_8_adaptation_"+adaptation_method+".csv"
    target_url="data/domain_adaptation_data/local_smote_8_adaptation_"+adaptation_method+".csv"
    
    sample_adaptation(source, target, source_url, target_url, adaptation_method=adaptation_method)
