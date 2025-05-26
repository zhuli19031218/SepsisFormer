# 导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.metrics import accuracy_score  # 分类正确率分数
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier 
import warnings
from utils_ml import *
import time
import os
from draw_roc import *
warnings.filterwarnings('ignore')
# 读取数据集
dataset = pd.read_csv('data/mimic34_smote_upsampling.csv')

# 划分训练集和测试集
# 去掉标签列
# 标准化
scaler=StandardScaler()
X = scaler.fit_transform(dataset.drop(['dead'],axis=1))
# 归一化
min_max_scaler=MinMaxScaler()
# X = min_max_scaler.fit_transform(dataset.drop(['dead'],axis=1))
# y为标签列
y = dataset['dead']
# 70%数据用于训练集，30%用于测试集
# X_train为训练集数据,X_test为测试集数据,y_train训练集标签,y_test为测试集标签
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# 模型定义
# estimators决策树个数，random_state控制随机状态
# 集成模型
dt = DecisionTreeClassifier(max_depth=100,min_samples_split=2,max_features = 10, random_state = 256)
# 模型训练
dt.fit(X_train, y_train)
# 评价指标
expected = y_test.astype('int')
precited = dt.predict(X_test).astype('int')
# AUC,ROC曲线下面积
rf_auc = metrics.roc_auc_score(y_test,dt.predict_proba(X_test)[:,1])
acc=accuracy_score(expected, precited)
tp, fp, tn, fn = compute_confusion_matrix(precited, expected)
# print(tp, fp, tn, fn,rf_auc,acc)
accuracy, precision, recall, f1 = compute_indexes(tp, fp, tn, fn)

print(f"{accuracy}\t{rf_auc}\t{precision}\t{recall}\t{f1}\t{tp}\t{fp}\t{tn}\t{fn}")
if not os.path.exists('logs'):
    os.mkdir('logs')
dir_name='决策树'+str(time.strftime("%Y%m%d-%H-%M", time.localtime()))+'smote_34'
logs_name = os.path.join("logs", dir_name)
if not os.path.exists(logs_name):
    os.mkdir(logs_name)
path=logs_name
s = open(os.path.join(path,"result.txt"), mode="a")
s.write(f"tp\tfp\ttn\tfn\n"
        f"{'%.4f' % tp}\t{'%.4f' % fp}\t{'%.4f' % tn}\t{'%.4f' % fn}\n"
        f"accuracy\tAUC\tprecision\trecall\tF1\n"
        f"{'%.4f' % accuracy}\t{'%.4f' % rf_auc}\t{'%.4f' % precision}\t{'%.4f' % recall}\t{'%.4f' % f1}")
# 绘制ROC曲线
draw_roc(y_test, dt.predict_proba(X_test)[:,1],'DT', os.path.join(path, './test_roc.pdf'))

