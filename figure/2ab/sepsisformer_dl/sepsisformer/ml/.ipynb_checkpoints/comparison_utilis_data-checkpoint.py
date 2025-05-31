import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw_ROC(fpr, tpr, auc, path):
    plt.figure(figsize=(8, 7), dpi=80, facecolor='w')  # dpi:每英寸长度的像素点数；facecolor 背景颜色
    plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  # 绘制刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)  # 绘制AUC 曲线
    plt.legend(loc='lower right')  # 设置显示标签的位置
    plt.xlabel('False Positive Rate', fontsize=14)  # 绘制x,y 坐标轴对应的标签
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.grid(b=True, ls=':')  # 绘制网格作为底板;b是否显示网格线；ls表示line style
    plt.title(u'TabTransformer ROC curve And  AUC', fontsize=18)  # 打印标题
    roc_image_path = os.path.join(path, "ROC.svg")
    plt.savefig(roc_image_path, format="svg")
    # plt.close('all')
    plt.show()

def stander_data(data, logs, stage):
    '''
    标准化
    '''
    # data1 = data.iloc[:, :18]
    # data2 = data.iloc[:, 18:]
    data1 = data.iloc[:, :8]
    data2 = data.iloc[:, 8]
    # print(data1[:10]).sample(frac=1)
    # print(data2[:10])
    if stage == 'pre':
        mean_std = pd.concat([pd.DataFrame(data1.mean(axis=0)), pd.DataFrame(data1.std(axis=0))], axis=1).T
        mean_std.to_csv(os.path.join(logs, ' mean_std.csv'), header=True, index=False)
        print('-----------\n', mean_std.T)
        # data1 = data1
        data1 = data1.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    if stage == 'post':
        mean_std = pd.read_csv(os.path.join(logs, ' mean_std.csv'))
        features_names = data1.columns
        assert (mean_std.shape[1] == len(features_names)), '训练集与全区数据的平均值、标准差形状不一致'
        for i in range(mean_std.shape[1]):
            data1[f'{features_names[i]}'] = data1[f'{features_names[i]}'].map(
                lambda x: (x - mean_std[f'{features_names[i]}'][0]) /mean_std[f'{features_names[i]}'][1])
    data = pd.concat([data1, data2], axis=1)
    # print(data[:10])
    return data

class TabDataset(Dataset):
    def __init__(self, data, target=None):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        _dict = {'data': torch.FloatTensor(data)}

        if self.target is not None:
            target = self.target[idx].item()
            _dict.update({'target': torch.tensor(target, dtype=torch.long)})
        #         """
        return _dict

# 读取原始的鄱阳湖数据
def get_data(logs, path:str, filtering=False):
    np.random.seed(256)
    pre_pyh_df = pd.read_csv(path).sample(frac=1)          # 原始数据，有表头
    standard_pre_pyh_df = stander_data(pre_pyh_df, logs=logs, stage='pre')

    original_pre_pyh_train_inputs = pre_pyh_df[:int(len(pre_pyh_df) * 0.7)]  # (4964, 43)->(5673, 42)
    original_pre_pyh_test_inputs = pre_pyh_df[int(len(pre_pyh_df) * 0.7):]  # (2128, 43)->(1419, 42)

    standard_pre_pyh_train_inputs = standard_pre_pyh_df[:int(len(pre_pyh_df) * 0.7)]
    standard_pre_pyh_test_inputs = standard_pre_pyh_df[int(len(pre_pyh_df) * 0.7):]

    pre_pyh_train_labels = standard_pre_pyh_train_inputs.pop('yfx').to_numpy()  # pd->>>numpy
    pre_pyh_test_labels = standard_pre_pyh_test_inputs.pop('yfx').to_numpy()    # pd->>>numpy

    print('标准化后训练集数据\n', standard_pre_pyh_train_inputs[:10])
    print('标准化后测试集数据\n', standard_pre_pyh_test_inputs[:10])
    standard_pre_pyh_train_inputs = standard_pre_pyh_train_inputs.to_numpy()  # pd->>>numpy
    standard_pre_pyh_test_inputs = standard_pre_pyh_test_inputs.to_numpy()    # pd->>>numpy

    print('原始训练集数据\n', original_pre_pyh_train_inputs[:10])
    print('原始测试集数据\n', original_pre_pyh_test_inputs[:10])
    original_pre_pyh_train_inputs = original_pre_pyh_train_inputs.iloc[:, :-1].to_numpy()  # pd->>>numpy
    original_pre_pyh_test_inputs = original_pre_pyh_test_inputs.iloc[:, :-1].to_numpy()  # pd->>>numpy

    if filtering:
        return standard_pre_pyh_train_inputs, pre_pyh_train_labels, \
               standard_pre_pyh_test_inputs, pre_pyh_test_labels,\
               original_pre_pyh_train_inputs, original_pre_pyh_test_inputs
    else:
        return standard_pre_pyh_train_inputs, pre_pyh_train_labels, \
               standard_pre_pyh_test_inputs, pre_pyh_test_labels

def pyh_dataloader(path:str, batch_size, logs):

    pre_pyh_train_inputs, pre_pyh_train_labels, pre_pyh_test_inputs, pre_pyh_test_labels = get_data(path, logs)
    pre_pyh_train_len, pre_pyh_test_len = len(pre_pyh_train_labels), len(pre_pyh_test_labels)
    print('type(pre_pyh_train_inputs)', pre_pyh_train_inputs.dtype)

    pre_pyh_train_factors = torch.FloatTensor(pre_pyh_train_inputs)   # torch.float32
    pre_pyh_test_factors = torch.FloatTensor(pre_pyh_test_inputs)     # torch.float32

    pre_pyh_train_dataset = TabDataset(data=pre_pyh_train_factors, target=pre_pyh_train_labels)
    pre_pyh_test_dataset = TabDataset(data=pre_pyh_test_factors, target=pre_pyh_test_labels)

    pre_pyh_train_dataset = DataLoader(pre_pyh_train_dataset, batch_size=batch_size, shuffle=False)
    pre_pyh_test_dataset = DataLoader(pre_pyh_test_dataset, batch_size=batch_size, shuffle=False)

    return pre_pyh_train_dataset, pre_pyh_test_dataset, pre_pyh_train_len, pre_pyh_test_len

def ml_get_dir_name(model_name):
    """
    :return: 生成一个以参数和时间戳命名的文件夹名,最终存放在log里
    """
    model_name = str(model_name)
    _time = str(time.strftime("%Y%m%d-%H-%M", time.localtime())) # 获取当前epoch的运行时刻
    dir_name = r'{}_'.format(model_name)
    dir_name = dir_name + _time

    return dir_name

def ml_mkdir(dir_name):
    """
    创建Logs、model文件夹，并以运行时间（年月日）+batch_size + epoch + Lr命名
    """
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('../model'):
        os.mkdir('../model')

    logs_name = os.path.join("logs", dir_name)
    if not os.path.exists(logs_name):
        os.mkdir(logs_name)

    model_name = os.path.join("../model", dir_name)
    if not os.path.exists(model_name):
        os.mkdir(model_name)

    return logs_name, model_name

def ml_save_file(data: pd.DataFrame, dir_name: str, file_name):
    """
    将pd格式数据写入指定文件夹
    """
    name = os.path.join(dir_name, file_name)       # name表示文件夹路径
    data.to_csv(name, index=False, header=False)   # 将pd格式数据写入’.CSV‘表格文件
    print(file_name, "save success!")

def ml_get_data(path:str, logs, stage):
    np.random.seed(256)
    pre_pyh_df = pd.read_csv(path).sample(frac=1)  # 原始数据，有表头

    standard_pre_pyh_df = stander_data(pre_pyh_df, logs=logs, stage=stage)
    # plt.imshow(standard_pre_pyh_df.corr())
    # sns.heatmap(standard_pre_pyh_df.corr(), cmap='YlGnBu', annot=True)
    # plt.show()
    original_pre_pyh_train_inputs = pre_pyh_df[:int(len(pre_pyh_df) * 0.4)]  # (4964, 43)->(5673, 42)
    original_pre_pyh_test_inputs = pre_pyh_df[int(len(pre_pyh_df) * 0.4):]  # (2128, 43)->(1419, 42)

    standard_pre_pyh_train_inputs = standard_pre_pyh_df[:int(len(pre_pyh_df) * 0.4)]
    standard_pre_pyh_test_inputs = standard_pre_pyh_df[int(len(pre_pyh_df) * 0.4):]

    pre_pyh_train_labels = standard_pre_pyh_train_inputs.pop('dead').to_numpy()  # pd->>>numpy
    pre_pyh_test_labels = standard_pre_pyh_test_inputs.pop('dead').to_numpy()
    # pre_pyh_train_labels = standard_pre_pyh_train_inputs.pop('yfx').to_numpy()  # pd->>>numpy
    # pre_pyh_test_labels = standard_pre_pyh_test_inputs.pop('yfx').to_numpy()  # pd->>>numpy

    print('标准化后训练集数据\n', standard_pre_pyh_train_inputs[:10])
    print('标准化后测试集数据\n', standard_pre_pyh_test_inputs[:10])
    standard_pre_pyh_train_inputs = standard_pre_pyh_train_inputs.to_numpy()  # pd->>>numpy
    standard_pre_pyh_test_inputs = standard_pre_pyh_test_inputs.to_numpy()  # pd->>>numpy

    print('原始训练集数据\n', original_pre_pyh_train_inputs[:10])
    print('原始测试集数据\n', original_pre_pyh_test_inputs[:10])
    original_pre_pyh_train_inputs = original_pre_pyh_train_inputs.iloc[:, :-1].to_numpy()  # pd->>>numpy
    original_pre_pyh_test_inputs = original_pre_pyh_test_inputs.iloc[:, :-1].to_numpy()  # pd->>>numpy

    return standard_pre_pyh_train_inputs, pre_pyh_train_labels, standard_pre_pyh_test_inputs, pre_pyh_test_labels

def whole_region_dataloader(path: str, batch_size, logs):
    original_whole_region_df = pd.read_csv(path)   # 全区数据有表头，无标签 .sample(frac=1)
    print('原始的全区数据\n', original_whole_region_df[:10])
    standard_whole_region_df = stander_data(original_whole_region_df, stage='post', logs=logs)
    print('标准化后的全区数据\n', standard_whole_region_df[:10])
    standard_whole_region_dataset = standard_whole_region_df.to_numpy()   # df->>numpy->tensor

    standard_whole_region_dataset = TabDataset(data=standard_whole_region_dataset, target=None)
    standard_whole_region_dataset = DataLoader(standard_whole_region_dataset, batch_size=batch_size, shuffle=False)

    return standard_whole_region_dataset, original_whole_region_df, standard_whole_region_df


def data_screening(standard_whole_features_prelabel_results_path,
                   original_whole_features_prelabel_results_path,
                   num_0, num_1, al1_filter=0.9, al0_filter=0.1,):
    """
    :param origion_pre_whole_region_concat_features_prelabel_results_path:
    :param num_0: 筛选补充0的数量
    :param num_1: 筛选补充1的数量
    :param al1_filter: 补充1的阈值
    :param al0_filter: 补充0的阈值
    :return:
    """
    np.random.seed(256)
    standard_whole_features_prelabel_results = pd.read_csv(standard_whole_features_prelabel_results_path, header=None).sample(frac=1)
    np.random.seed(256)
    original_whole_features_prelabel_results = pd.read_csv(original_whole_features_prelabel_results_path, header=None).sample(frac=1)
    # print(standard_whole_features_prelabel_results[:10])
    # print(original_whole_features_prelabel_results[:10])

    standard_whole_features_prelabel_results.columns = [i for i in range(
        len(standard_whole_features_prelabel_results.columns))]
    original_whole_features_prelabel_results.columns = [i for i in range(
        len(original_whole_features_prelabel_results.columns))]

    #  分成两部分的原因是被A选中作为标签1的数据，不能被B选中作为标签0
    iter = int(standard_whole_features_prelabel_results.shape[0]/2)
    standard_whole_features_prelabel_results_part1 = standard_whole_features_prelabel_results[:iter]
    standard_whole_features_prelabel_results_part2 = standard_whole_features_prelabel_results[iter:]

    original_whole_features_prelabel_results_part1 = original_whole_features_prelabel_results[:iter]
    original_whole_features_prelabel_results_part2 = original_whole_features_prelabel_results[iter:]

    # ------------------------------------------------------#
    # n：用来指定随机抽取的样本数目（行数目）或者列数目
    # replace:False表示执行无放回抽样，True表示执行有放回抽样
    # random_state:设置随机数种子,这个参数可以复现抽样结果
    # axis=:对行进行抽样，axis=1:对列进行抽样
    # ------------------------------------------------------#
    # -1 指的是最后一列，即预测为滑坡的概率, 预测概率大于阈值1的数据
    # 从筛选后的全区数据中，只返回一部分

    standard_whole_features_prelabel_results_1 = standard_whole_features_prelabel_results_part1[
        (standard_whole_features_prelabel_results_part1.iloc[:, -1] > al0_filter)]  # al1_filter
    original_whole_features_prelabel_results_1 = original_whole_features_prelabel_results_part1[
        (original_whole_features_prelabel_results_part1.iloc[:, -1] > al0_filter)]  # al1_filter

    standard_whole_features_prelabel_results_0 = standard_whole_features_prelabel_results_part2[
        (standard_whole_features_prelabel_results_part2.iloc[:, -1] < al1_filter)]  # al0_filter
    original_whole_features_prelabel_results_0 = original_whole_features_prelabel_results_part2[
        (original_whole_features_prelabel_results_part2.iloc[:, -1] < al1_filter)]  # al0_filter

    # 保证origion与pre随即返回的值一致
    random_num_1 = np.random.randint(0, 999)  # 生成一个指定范围内的整数
    # print('random_num', random_num_1)
    np.random.seed(random_num_1)
    standard_whole_features_prelabel_results_1 = standard_whole_features_prelabel_results_1.sample(
        n=num_1, replace=True, random_state=None, axis=0)
    np.random.seed(random_num_1)
    original_whole_features_prelabel_results_1 = original_whole_features_prelabel_results_1.sample(
        n=num_1, replace=True, random_state=None, axis=0)

    # print('standard_whole_features_prelabel_results_1', '\n',
    #       standard_whole_features_prelabel_results_1[:10])
    # print('original_whole_features_prelabel_results_1 ', '\n',
    #       original_whole_features_prelabel_results_1[:10])

    random_num_2 = np.random.randint(0, 999)  # 生成一个指定范围内的整数
    np.random.seed(random_num_2)
    standard_whole_features_prelabel_results_0 = standard_whole_features_prelabel_results_0.sample(
        n=num_0, replace=True, random_state=None, axis=0)
    np.random.seed(random_num_2)
    original_whole_features_prelabel_results_0 = original_whole_features_prelabel_results_0.sample(
        n=num_0, replace=True, random_state=None, axis=0)

    # print('standard_whole_features_prelabel_results_0', '\n',
    #       standard_whole_features_prelabel_results_0[:10])
    # print('original_whole_features_prelabel_results_0', '\n',
    #       original_whole_features_prelabel_results_0[:10])

    standard_whole_features_1 = standard_whole_features_prelabel_results_1.iloc[:, :-3]  # 只取出前21个特性因子
    standard_whole_features_0 = standard_whole_features_prelabel_results_0.iloc[:, :-3]  # 只取出前21个特性因子

    original_whole_features_1 = original_whole_features_prelabel_results_1.iloc[:, :-3]  # 只取出前21个特性因子
    original_whole_features_0 = original_whole_features_prelabel_results_0.iloc[:, :-3]  # 只取出前21个特性因子

    standard_whole_features_1.columns = [i for i in range(standard_whole_features_1.shape[1])]
    original_whole_features_1.columns = [i for i in range(original_whole_features_1.shape[1])]

    standard_whole_features_1[standard_whole_features_1.shape[1]] = 1   # 添加标签列，21个特性因子+1个真实标签,包含表头
    original_whole_features_1[original_whole_features_1.shape[1]] = 1   # 添加标签列，21个特性因子+1个真实标签,包含表头

    standard_whole_features_label_1 = standard_whole_features_1
    original_whole_features_label_1 = original_whole_features_1

    standard_whole_features_0.columns = [i for i in range(standard_whole_features_0.shape[1])]
    original_whole_features_0.columns = [i for i in range(original_whole_features_0.shape[1])]

    standard_whole_features_0[standard_whole_features_0.shape[1]] = 0  # 添加标签列，21个特性因子+1个真实标签,包含表头
    original_whole_features_0[original_whole_features_0.shape[1]] = 0  # 添加标签列，21个特性因子+1个真实标签,包含表头

    standard_whole_features_label_0 = standard_whole_features_0
    original_whole_features_label_0 = original_whole_features_0

    # 包含表头
    return standard_whole_features_label_1, standard_whole_features_label_0,\
           original_whole_features_label_1, original_whole_features_label_0

def data_process(pre_pyh_train_results_path,
                 pre_pyh_test_results_path,
                 standard_pre_pyh_train_features,
                 original_pre_pyh_train_features,
                 pre_pyh_train_labels,
                 standard_pre_pyh_test_features,
                 original_pre_pyh_test_features,
                 pre_pyh_test_labels,
                 standard_whole_features_prelabel_results_path,
                 original_whole_features_prelabel_results_path,
                 threshold=0.6,
                 pre_pyh_save_dir=None):

    # print('pre_pyh_train_features.shape:', standard_pre_pyh_train_features.shape)  # train_inputs: (4964, 21)
    # print('pre_pyh_train_labels.shape:', pre_pyh_train_labels.shape)    # train_label: torch.Size((4964）])
    # print('pre_pyh_test_labels.shape:', pre_pyh_test_labels.shape)      # test_label: torch.Size([2128])
    # print('pre_pyh_test_features.shape:', standard_pre_pyh_test_features.shape)    # torch.Size(2128, 21)

    # -------------------------------------------------------#
    # 只取一半是因为原始数据中，一半标签为1的样本与真实标签为0的样本各占一半
    # 使用sample()方法返回的训练集中、验证集中标签为1、0的数据各占一半
    # -------------------------------------------------------#

    # 训练集的预测结果,不包含表头，虽有索引项，但其索引不算列->>>将pd格式转换为numpy数组格式，不包含表头和索引
    # 测试集的预测结果,不包含表头，虽有索引项，但其索引不算列->>>将pd格式转换为numpy数组格式，不包含表头和索引
    pre_pyh_train_results = np.array(pd.read_csv(pre_pyh_train_results_path, header=None))
    pre_pyh_test_results = np.array(pd.read_csv(pre_pyh_test_results_path, header=None))

    assert (original_pre_pyh_train_features.shape[0] == standard_pre_pyh_train_features.shape[0] ==
            pre_pyh_train_labels.shape[0] == pre_pyh_train_results.shape[0]), "测试集预测结果、特征值、标签形状不一致"
    assert (original_pre_pyh_test_features.shape[0] == standard_pre_pyh_test_features.shape[0] ==
            pre_pyh_test_labels.shape[0] == pre_pyh_test_results.shape[0]), "训练集预测结果、特征值、标签形状不一致"

    # np.c_是按列连接两个矩阵，就是把两矩阵左右相加，要求行数相等
    # numpy->>>pd
    pre_pyh_train_features_labels_results = pd.DataFrame(np.c_[standard_pre_pyh_train_features, pre_pyh_train_labels, pre_pyh_train_results])
    pre_pyh_test_features_labels_results  = pd.DataFrame(np.c_[standard_pre_pyh_test_features, pre_pyh_test_labels, pre_pyh_test_results])
    print(pre_pyh_train_features_labels_results [:10])
    print(pre_pyh_test_features_labels_results[:10])
    # print('筛选前训练集的形状:', pre_pyh_train_concat_features_labels_results.shape)        # 筛选前训练集的长度: (5673, 24)
    # print('筛选前测试集的形状:', pre_pyh_test_concat_features_labels_results.shape)         # 筛选前测试集的长度: (1419, 24)

    original_pre_pyh_train_features_labels_results = pd.DataFrame(np.c_[original_pre_pyh_train_features, pre_pyh_train_labels, pre_pyh_train_results])
    original_pre_pyh_test_features_labels_results  = pd.DataFrame(np.c_[original_pre_pyh_test_features, pre_pyh_test_labels, pre_pyh_test_results])
    print('original_pre_pyh_train_features_labels_results\n', original_pre_pyh_train_features_labels_results[:10])
    print('original_pre_pyh_test_features_labels_results\n', original_pre_pyh_test_features_labels_results[:10])

    # mode="a": 打开一个文件用于追加,不会清空文件内容，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
    # 写入内容：----------------------------------------2022-06-07 21:03:08-----------------------------------------
    s = open(os.path.join(pre_pyh_save_dir, "screening.txt"), mode="a")
    s.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())).center(100, "-") + "\n")

    # 偏移量
    # 添加偏度
    # 使用 DataFrame的 skew() 方法来计算所有数据属性的高斯分布偏离情况。
    # skew() 函数的结果显示了数据分布是左偏还是右偏。当数据接近0时，表示数据的偏差非常小
    print('开始自筛选,计算阈值'.center(100, '-'))
    # pre_pyh_train_concat_features_labels_results: 42个特性因子 + 1个真实标签 + 2个预测结果 (-1:指的是最后一列，即预测为滑坡的概率)
    pre_pyh_concat_train_test = pd.concat([pre_pyh_train_features_labels_results.iloc[:, -1],
                                           pre_pyh_test_features_labels_results.iloc[:, -1]])   # (5673+1419,)=(7092,)
    sk = pre_pyh_concat_train_test.skew()
    sk = sk / (1 + 2 * abs(sk))
    T1 = abs(threshold * sk) ** 0.5
    T0 = 1 - T1
    print(f"阈值：{threshold}, T1:{T1} T0:{T0}")         # T1:0.3957027537424594
    s.write(f"阈值：{threshold}, T1:{T1} T0:{T0}\n")       # T0:0.6042972462575407

    # 筛选(删除)训练集
    # 只保留标签为1时，预测结果大于某一阈值的数据，以及标签为0时，预测结果低于另外一个阈值的数据
    # -1:指的是最后一列，即预测为滑坡的概率
    # -3:指的是倒数第三列，即真实标签对应的列
    # |:代表或
    # pre_pyh_train_concat_features_labels_part1(3971, 24)
    # pre_pyh_train_concat_features_labels_part2(1702, 24)
    train_threshold = int(len(pre_pyh_train_features_labels_results)*0.7)  # int(5673*0.7)=3971
    pre_pyh_train_features_labels_results_part1 = pre_pyh_train_features_labels_results.iloc[:train_threshold]
    pre_pyh_train_features_labels_results_part2 = pre_pyh_train_features_labels_results.iloc[train_threshold:]

    # original_pre_pyh_train_features_labels_results_part1 = original_pre_pyh_train_features_labels_results.iloc[:train_threshold]
    # original_pre_pyh_train_features_labels_results_part2 = original_pre_pyh_train_features_labels_results.iloc[train_threshold:]

    # pandas 计数函数value_counts()
    train_cate_dict = dict(pre_pyh_train_features_labels_results_part1[21].value_counts())  # 21对应的是标签列
    train_cate_array = []
    for k, v in train_cate_dict.items():
        train_cate_array.append(v)
    print(train_cate_dict)
    # print(train_cate_array)
    # print(dict(pre_pyh_train_concat_features_labels_results_part1[21].value_counts()))

    filtering_pyh_train_features_labels_results = pre_pyh_train_features_labels_results_part1[
        ((pre_pyh_train_features_labels_results_part1.iloc[:, -1] > T1) &
         (pre_pyh_train_features_labels_results_part1.iloc[:, -3] == 1)) |
        ((pre_pyh_train_features_labels_results_part1.iloc[:, -1] < T0) &
         (pre_pyh_train_features_labels_results_part1.iloc[:, -3] == 0))]

    # original_pre_pyh_train_features_labels_results = original_pre_pyh_train_features_labels_results_part1[
    #     ((original_pre_pyh_train_features_labels_results_part1.iloc[:, -1] > T1) &
    #      (original_pre_pyh_train_features_labels_results_part1.iloc[:, -3] == 1)) |
    #     ((original_pre_pyh_train_features_labels_results_part1.iloc[:, -1] < T0) &
    #      (original_pre_pyh_train_features_labels_results_part1.iloc[:, -3] == 0))]

    # 筛选(删除)训练集
    # 只保留标签为1时，预测结果大于某一阈值的数据，以及标签为0时，预测结果低于另外一个阈值的数据
    # -1:指的是最后一列，即预测为滑坡的概率
    # -3:指的是倒数第三列，即真实标签对应的列
    # pre_pyh_test_concat_features_labels_part1(993, 24)
    # pre_pyh_test_concat_features_labels_part2(426, 24)
    test_threshold = int(len(pre_pyh_test_features_labels_results) * 0.7)
    pre_pyh_test_features_labels_results_part1 = pre_pyh_test_features_labels_results.iloc[:test_threshold]
    pre_pyh_test_features_labels_results_part2 = pre_pyh_test_features_labels_results.iloc[test_threshold:]

    original_pre_pyh_test_features_labels_results_part1 = original_pre_pyh_test_features_labels_results.iloc[:test_threshold]
    original_pre_pyh_test_features_labels_results_part2 = original_pre_pyh_test_features_labels_results.iloc[test_threshold:]

    test_cate_dict = dict(pre_pyh_test_features_labels_results_part1[21].value_counts())
    test_cate_array = []
    for k, v in test_cate_dict.items():
        test_cate_array.append(v)
    # print(test_cate_array)
    # print(dict(pre_pyh_test_concat_features_labels_results_part1[21].value_counts()))
    filtering_pyh_test_features_labels_results = pre_pyh_test_features_labels_results_part1[
        ((pre_pyh_test_features_labels_results_part1.iloc[:, -1] > T1) &
         (pre_pyh_test_features_labels_results_part1.iloc[:, -3] == 1)) |
        ((pre_pyh_test_features_labels_results_part1.iloc[:, -1] < T0) &
         (pre_pyh_test_features_labels_results_part1.iloc[:, -3] == 0))]

    original_pre_pyh_test_features_labels_results = original_pre_pyh_test_features_labels_results_part1[
        ((original_pre_pyh_test_features_labels_results_part1.iloc[:, -1] > T1) &
         (original_pre_pyh_test_features_labels_results_part1.iloc[:, -3] == 1)) |
        ((original_pre_pyh_test_features_labels_results_part1.iloc[:, -1] < T0) &
         (original_pre_pyh_test_features_labels_results_part1.iloc[:, -3] == 0))]

    # print('剔除部分数据后训练集的形状:', filtering_pyh_train_concat_features_labels_results.shape)
    # print('剔除部分数据后测试集的形状:', filtering_pyh_test_concat_features_labels_results.shape)
    s.write(f'删除部分数据后，训练集数据量：{filtering_pyh_train_features_labels_results.shape}\n')
    s.write(f'剔除部分数据后，测试集数据量：{filtering_pyh_test_features_labels_results.shape}\n')

    # ----------------------------------------------------------------#
    # 删除训练集的数量
    # 只取一半是因为原始数据中，一半标签为1的样本与真实标签为0的样本各占一半
    # train_num_1：训练集经过自筛选后需要补充标签为1的数据的数量
    # train_num_0：训练集经过自筛选后需要补充标签为0的数据的数量
    # num_origin_train=10752/2=5376
    # 42为真实标签对应的列
    # 使用sample()方法返回的训练集中、验证集中标签为1、0的数据各占一半
    # ----------------------------------------------------------------#
    train_num_1 = train_cate_array[0] - filtering_pyh_train_features_labels_results[
        filtering_pyh_train_features_labels_results.iloc[:, -3] == 1].shape[0]
    train_num_0 = train_cate_array[1] - filtering_pyh_train_features_labels_results[
        filtering_pyh_train_features_labels_results.iloc[:, -3] == 0].shape[0]

    s.write(f"训练集需要补充标签为1的数据数量：{train_num_1}, \n训练集需要补充标签为0的数据数量：{train_num_0}\n")

    # ----------------------------------------------------------------#
    # 删除测试集的数量
    # 只取一半是因为原始数据中，一半标签为1的样本与真实标签为0的样本各占一半
    # test_num_1：训练集经过自筛选后需要补充标签为1的数据的数量
    # test_num_0：训练集经过自筛选后需要补充标签为0的数据的数量
    # num_origin_test=2688/2=1344
    # 42为真实标签对应的列
    # 使用sample()方法返回的训练集中、验证集中标签为1、0的数据各占一半
    # ----------------------------------------------------------------#
    test_num_1 = test_cate_array[0] - filtering_pyh_test_features_labels_results[
        filtering_pyh_test_features_labels_results.iloc[:, -3] == 1].shape[0]
    test_num_0 = test_cate_array[1] - filtering_pyh_test_features_labels_results[
        filtering_pyh_test_features_labels_results.iloc[:, -3] == 0].shape[0]

    s.write(f"测试集需要补充标签为1的数据数量：{test_num_1}, \n测试集需要补充标签为0的数据数量：{test_num_0}\n")

    # 取出特性因子+真实标签
    filtering_pyh_train_features_labels = filtering_pyh_train_features_labels_results.iloc[:, :-2]
    filtering_pyh_test_features_labels = filtering_pyh_test_features_labels_results.iloc[:, :-2]
    original_pyh_test_features_labels = original_pre_pyh_test_features_labels_results.iloc[:, :-2]

    print("data_screening".center(100, "-"))
    standard_supplement_total_1, standard_supplement_total_0, original_supplement_total_1, original_supplement_total_0 = \
        data_screening(standard_whole_features_prelabel_results_path=standard_whole_features_prelabel_results_path,
                       original_whole_features_prelabel_results_path=original_whole_features_prelabel_results_path,
                       num_0=train_num_0+test_num_0,
                       num_1=train_num_1+test_num_1)  # al0_filter=T1, al1_filter=T0

    standard_supplement_train_addition_0 = standard_supplement_total_0.iloc[:train_num_0]  # 训练集添加标签为0的数据
    standard_supplement_test_addition_0 = standard_supplement_total_0.iloc[train_num_0:test_num_0 + train_num_0, :]  # 测试集添加标签为0的数据
    original_supplement_test_addition_0 = original_supplement_total_0.iloc[train_num_0:test_num_0 + train_num_0, :]  # 测试集添加标签为0的数据
    standard_supplement_train_addition_1 = standard_supplement_total_1.iloc[:train_num_1]  # 训练集添加标签为1的数据
    standard_supplement_test_addition_1 = standard_supplement_total_1.iloc[train_num_1:test_num_1 + train_num_1, :]  # 测试集添加标签为1的数据
    original_supplement_test_addition_1 = original_supplement_total_1.iloc[train_num_1:test_num_1 + train_num_1, :]  # 测试集添加标签为1的数据

    # print('supplement_train_addition_1.shape', supplement_train_addition_1.shape)
    # print('supplement_train_addition_0.shape', supplement_train_addition_0.shape)
    # print('supplement_test_addition_1.shape', supplement_test_addition_1.shape)
    # print('supplement_test_addition_0.shape', supplement_test_addition_0.shape)

    standard_supplement_total_train = pd.concat([standard_supplement_train_addition_0, standard_supplement_train_addition_1], axis=0)
    standard_supplement_total_test = pd.concat([standard_supplement_test_addition_0, standard_supplement_test_addition_1], axis=0)
    original_supplement_total_test = pd.concat([original_supplement_test_addition_0, original_supplement_test_addition_1], axis=0)

    standard_post_pyh_train_total_features_labels = pd.concat([filtering_pyh_train_features_labels, standard_supplement_total_train])
    standard_post_pyh_test_total_features_labels = pd.concat([filtering_pyh_test_features_labels, standard_supplement_total_test])
    original_post_pyh_test_total_features_labels = pd.concat([original_pyh_test_features_labels, original_supplement_total_test])

    s.write(f"补充部分后，训练集数据量：{standard_post_pyh_train_total_features_labels.shape}\n")  # 补充后，训练集数据量： (10752, 43)
    s.write(f"补充部分后，测试集数据量：{standard_post_pyh_test_total_features_labels.shape}\n")  # test_inputs.shape:  (2688, 42)
    s.write(f"补充部分后，original测试集数据量：{original_post_pyh_test_total_features_labels.shape}\n")  # test_inputs.shape:  (2688, 42)

    standard_pre_pyh_train_features_labels_part2 = pre_pyh_train_features_labels_results_part2.iloc[:, :-2]
    standard_pre_pyh_test_features_labels_part2 = pre_pyh_test_features_labels_results_part2.iloc[:, :-2]
    original_pre_pyh_test_features_labels_part2 = original_pre_pyh_test_features_labels_results_part2.iloc[:, :-2]

    standard_post_pyh_train_total_features_labels = pd.concat([standard_post_pyh_train_total_features_labels,
                                                               standard_pre_pyh_train_features_labels_part2])
    standard_post_pyh_test_total_features_labels = pd.concat([standard_post_pyh_test_total_features_labels,
                                                              standard_pre_pyh_test_features_labels_part2])
    original_post_pyh_test_total_features_labels = pd.concat([original_post_pyh_test_total_features_labels,
                                                              original_pre_pyh_test_features_labels_part2])

    # print("筛选、补充、合并后，最终训练集数据量：", post_pyh_train_total_data.shape,
    #       "\n筛选、补充、合并后，最终测试集数据量：", post_pyh_test_total_data.shape)
    s.write(f"筛选、补充、合并后，standard_post_pyh_train_total_features_labels：{standard_post_pyh_train_total_features_labels.shape}\n")  # 补充后，训练集数据量： (10752, 43)
    s.write(f"筛选、补充、合并后，standard_post_pyh_test_total_features_labels：{standard_post_pyh_test_total_features_labels.shape}\n")  # test_inputs.shape:  (2688, 42)
    s.write(f"筛选、补充、合并后，original_post_pyh_test_total_features_labels：{original_post_pyh_test_total_features_labels.shape}\n")
    s.close()

    standard_post_pyh_train_total_features = standard_post_pyh_train_total_features_labels.iloc[:, :-1]                # 只取出测试集的前42个特性因子
    standard_post_pyh_test_total_features = standard_post_pyh_test_total_features_labels.iloc[:, :-1]
    original_post_pyh_test_total_features = original_post_pyh_test_total_features_labels.iloc[:, :-1]

    standard_post_pyh_train_total_labels = standard_post_pyh_train_total_features_labels.iloc[:, -1]  # 取出测试集的标签
    standard_post_pyh_test_total_labels = standard_post_pyh_test_total_features_labels.iloc[:, -1]    # 取出测试集的标签
    original_post_pyh_test_total_labels = original_post_pyh_test_total_features_labels.iloc[:, -1]

    for i in range(len(standard_post_pyh_test_total_labels)):
        # print(i)
        # print(i, 'standard_post_pyh_test_total_labels[i]', standard_post_pyh_test_total_labels[i])
        # print(i, 'original_post_pyh_test_total_labels[i]', original_post_pyh_test_total_labels[i])
        assert (np.array(standard_post_pyh_test_total_labels)[i] == np.array(original_post_pyh_test_total_labels)[i]), \
            f"('np.array(standard_post_pyh_test_total_labels)[i] == np.array(original_post_pyh_test_total_labels)[i]')"
    assert (standard_pre_pyh_train_features.shape == standard_post_pyh_train_total_features.shape), "筛选前后训练集数据维度不一致"
    assert (standard_pre_pyh_test_features.shape == standard_post_pyh_test_total_features.shape), "筛选前后测试集数据维度不一致"

    return standard_post_pyh_train_total_features, standard_post_pyh_train_total_labels, \
           standard_post_pyh_test_total_features, standard_post_pyh_test_total_labels,\
           original_post_pyh_test_total_features, original_post_pyh_test_total_labels
