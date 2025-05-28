import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
class NMTCritierion(nn.Module):
    """
    1. Add label smoothing
    标签平滑
    """
    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax()

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            # print('one_hot', one_hot[:10])
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
            # print('gtruth',  gtruth[:10])
        loss = self.criterion(scores, gtruth)
        return loss



def stander_data(data):
    '''
    标准化
    :return:
    '''
   

    data = data.iloc[:, :]
    data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

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

        # data = self.data[idx]
        # # # print(type(data))  #<class 'numpy.ndarray'>
        # _dict = {'data': torch.tensor(data, dtype=torch.float)}

        if self.target is not None:
            target = self.target[idx].item()
            _dict.update({'target': torch.tensor(target, dtype=torch.float)})
        return _dict
# 读取原始数据
def get_data1(path: str, logs):
    np.random.seed(256)
    pre_pyh_df = pd.read_csv(path).sample(frac=1)  # 原始数据，有表头
    pre_pyh_label = pre_pyh_df.pop('dead').to_numpy()
    pre_pyh_train_labels = torch.LongTensor(pre_pyh_label)  # numpy->>>torch.tensor,int64
    original_pre_pyh_train_inputs = pre_pyh_df  # (4964, 43)->(5673, 42)
    # 保存原始训练集、验证集、测试集数据
    original_pre_pyh_train_inputs.to_csv(os.path.join(logs, 'original_pre_pyh_train_features.csv'), index=False)
 
    standard_pre_pyh_train_inputs = stander_data(original_pre_pyh_train_inputs).to_numpy()

    return standard_pre_pyh_train_inputs, pre_pyh_train_labels

def pyh_dataloader1(path:str, logs):
    pre_pyh_train_factors, pre_pyh_train_labels = get_data1(path, logs)

    pre_pyh_train_factors = torch.FloatTensor(pre_pyh_train_factors)  # torch.float32

    return pre_pyh_train_factors, pre_pyh_train_labels

# 读取原始数据
def get_data(path: str, logs,split):
    np.random.seed(256)
    pre_pyh_df = pd.read_csv(path).sample(frac=1)  # 原始数据，有表头
    pre_pyh_label = pre_pyh_df.pop('dead').to_numpy()
    pre_pyh_train_labels = torch.LongTensor(pre_pyh_label[:int(len(pre_pyh_df) * split)])  # numpy->>>torch.tensor,int64
    pre_pyh_test_labels = torch.LongTensor(pre_pyh_label[int(len(pre_pyh_df) * split):])  # numpy->>>torch.tensor,int64
    original_pre_pyh_train_inputs = pre_pyh_df[:int(len(pre_pyh_df) * split)]  # (4964, 43)->(5673, 42)
    original_pre_pyh_test_inputs = pre_pyh_df[int(len(pre_pyh_df) * split):]  # (2128, 43)->(1419, 42)
    # 保存原始训练集、验证集、测试集数据
    original_pre_pyh_train_inputs.to_csv(os.path.join(logs, 'original_pre_pyh_train_features.csv'), index=False)
    original_pre_pyh_test_inputs.to_csv(os.path.join(logs, 'original_pre_pyh_test_features.csv'), index=False)
    # standard_pre_pyh_train_inputs = stander_data(original_pre_pyh_train_inputs).to_numpy()
    # standard_pre_pyh_test_inputs = stander_data(original_pre_pyh_test_inputs).to_numpy()
    standard_pre_pyh_train_inputs = original_pre_pyh_train_inputs.to_numpy()
    standard_pre_pyh_test_inputs = original_pre_pyh_test_inputs.to_numpy()
    return standard_pre_pyh_train_inputs, pre_pyh_train_labels, standard_pre_pyh_test_inputs, pre_pyh_test_labels

def pyh_dataloader(path:str, logs,split):
    pre_pyh_train_factors, pre_pyh_train_labels, pre_pyh_test_factors, \
    pre_pyh_test_labels = get_data(path, logs,split)

    pre_pyh_train_factors = torch.FloatTensor(pre_pyh_train_factors)  # torch.float32
    pre_pyh_test_factors = torch.FloatTensor(pre_pyh_test_factors)  # torch.float32

    return pre_pyh_train_factors, pre_pyh_train_labels, pre_pyh_test_factors, pre_pyh_test_labels
