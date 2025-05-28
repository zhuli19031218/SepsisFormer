import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")
class AnyuanDataset(Dataset):
    """
            数据集类
    """

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


def get_data(path: str, logs='./logs'):
    """
        获取数据，path：数据路径，logs：日志路径
        返回数据 (标准化后的训练集，训练集标签，标准化后的测试集，测试集标签)
    """
    np.random.seed(256) #记得默认修改回256
    pre_ay_df = pd.read_csv(path).sample(frac=1)  # 原始数据，有表头
    # 去除object id
    # pre_ay_df.pop("SOFAscore")
    length_all = int(len(pre_ay_df) * 0.7)

    inputs = pre_ay_df[:]
    inputs.to_csv(os.path.join(logs, 'features.csv'), index=False)
    test_inputs = pre_ay_df[int(len(pre_ay_df) * 0.7):]
    test_inputs.to_csv(os.path.join(logs, 'test_features.csv'), index=False)

    # # original_pre_ay_train_inputs = pre_ay_df[:length_all]  # (4964, 43)->(5673, 42)
    # original_pre_ay_test_inputs = pre_ay_df[length_all:]  # (2128, 43)->(1419, 42)
    #
    # # print('原始训练集数据\n', original_pre_ay_train_inputs[:10])
    # # print('原始测试集数据\n', original_pre_ay_test_inputs[:10])
    # # 原始数据的备份吗？好像没啥用
    # # original_pre_ay_train_inputs.to_csv(os.path.join(logs, 'original_pre_ay_train_features.csv'), index=False)
    # original_pre_ay_test_inputs.to_csv(os.path.join(logs, 'original_pre_test_data.csv'), index=False)

    # 获取标签
    pre_ay_label = pre_ay_df.pop("dead").to_numpy()  # susceptibility  yfx

    # 按7：3划分训练集和测试集

    pre_ay_train_labels = torch.LongTensor(pre_ay_label[:length_all])  # numpy->>>torch.tensor,int64
    pre_ay_test_labels = torch.LongTensor(pre_ay_label[length_all:])  # numpy->>>torch.tensor,int64

    original_pre_pyh_train_inputs = pre_ay_df[:int(len(pre_ay_df) * 0.7)]
    original_pre_pyh_test_inputs = pre_ay_df[int(len(pre_ay_df) * 0.7):]

    # original_pre_pyh_train_inputs = pre_ay_df[:int(len(pre_ay_df) * 0.7)].to_numpy()  # (4964, 43)->(5673, 42)
    # original_pre_pyh_test_inputs = pre_ay_df[int(len(pre_ay_df) * 0.7):].to_numpy()   # (2128, 43)->(1419, 42)

    print('原始训练集数据\n', original_pre_pyh_train_inputs[:10])
    print('原始测试集数据\n', original_pre_pyh_test_inputs[:10])
    original_pre_pyh_train_inputs.to_csv(os.path.join(logs, 'original_pre_pyh_train_features.csv'), index=False)
    original_pre_pyh_test_inputs.to_csv(os.path.join(logs, 'original_pre_pyh_test_features.csv'), index=False)

    # 进行标准化处理 0 均值 1方差
    # standard_pre_ay_df = stander_data(pre_ay_df)
    # standard_pre_ay_train_inputs = standard_pre_ay_df[:length_all].to_numpy()
    # standard_test_data = standard_pre_ay_df[length_all:]
    # standard_pre_ay_test_inputs = standard_test_data.to_numpy()

    standard_pre_ay_train_inputs = pre_ay_df[:length_all].to_numpy()
    standard_pre_ay_test_inputs = pre_ay_df[length_all:].to_numpy()

    # pd.concat(standard_test_data, pre_ay_df.pop("dead"))
    # print('标准化后训练集数据\n', standard_pre_ay_df[:length_all][:10])
    # print('标准化后测试集数据\n', standard_pre_ay_df[length_all:][:10])
    # standard_pre_ay_df.to_csv(os.path.join(logs, 'standard_pre_ay_features.csv'), index=False)


    return standard_pre_ay_train_inputs, pre_ay_train_labels, standard_pre_ay_test_inputs, pre_ay_test_labels
    # return original_pre_pyh_train_inputs, pre_ay_train_labels, original_pre_pyh_test_inputs, pre_ay_test_labels

def get_data0():
    X_train = pd.read_csv('I:\mimic_transformer\data0\X_train.csv')
    y_train = pd.read_csv('I:\mimic_transformer\data0\y_train.csv')
    X_test = pd.read_csv('I:\mimic_transformer\data0\X_test.csv')
    y_test = pd.read_csv('I:\mimic_transformer\data0\y_test.csv')

    y_train = torch.LongTensor(y_train.to_numpy())
    y_test = torch.LongTensor(y_test.to_numpy())
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    return X_train, y_train, X_test, y_test
def get_data1(path1: str, logs='./logs'):
    np.random.seed(256)
    pre_ay_df = pd.read_csv(path1).sample(frac=1)  # 原始数据，有表头
    length_all = int(len(pre_ay_df))
    # 获取标签
    pre_ay_label = pre_ay_df.pop("dead").to_numpy()  # susceptibility  yfx
    pre_ay_val_labels = torch.LongTensor(pre_ay_label[:])  # numpy->>>torch.tensor,int64
    # original_pre_pyh_val_inputs = pre_ay_df[:]  # (2128, 43)->(1419, 42)

    original_pre_pyh_val_inputs = pre_ay_df[:].to_numpy()  # (2128, 43)->(1419, 42)

    # original_pre_pyh_val_inputs.to_csv(os.path.join(logs, 'original_pre_pyh_val_features.csv'), index=False)
    # standard_pre_ay_df = stander_data(pre_ay_df)
    # standard_pre_ay_val_inputs = standard_pre_ay_df[:].to_numpy()

    # return standard_pre_ay_val_inputs, pre_ay_val_labels
    return original_pre_pyh_val_inputs, pre_ay_val_labels

def stander_data(data):
    """
        数据集标准化
    """
    data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    return data


def ay_dataloader(path: str, batch_size, logs):
# def ay_dataloader(path: str, path1: str, batch_size, logs): #
    """
        获取数据集与标签
    """

    pre_ay_train_inputs, pre_ay_train_labels, pre_ay_test_inputs, pre_ay_test_labels = get_data(path, logs)
    # pre_ay_val_inputs, pre_ay_val_labels = get_data1(path1, logs)
    pre_ay_train_len, pre_ay_test_len = len(pre_ay_train_labels), len(pre_ay_test_labels)
    # pre_ay_val_len = len(pre_ay_val_labels)

    pre_ay_train_factors = torch.FloatTensor(pre_ay_train_inputs)  # torch.float32
    pre_ay_test_factors = torch.FloatTensor(pre_ay_test_inputs)  # torch.float32
    # pre_ay_val_factors = torch.FloatTensor(pre_ay_val_inputs)

    pre_ay_train_dataset = AnyuanDataset(data=pre_ay_train_factors, target=pre_ay_train_labels)
    pre_ay_test_dataset = AnyuanDataset(data=pre_ay_test_factors, target=pre_ay_test_labels)
    # pre_ay_val_dataset = AnyuanDataset(data=pre_ay_val_factors, target=pre_ay_val_labels)

    pre_ay_train_dataset = DataLoader(pre_ay_train_dataset, batch_size=batch_size, shuffle=False)
    pre_ay_test_dataset = DataLoader(pre_ay_test_dataset, batch_size=batch_size, shuffle=False)
    # pre_ay_val_dataset = DataLoader(pre_ay_val_dataset, batch_size=batch_size, shuffle=False)

    # return pre_ay_train_dataset, pre_ay_test_dataset, pre_ay_train_len, pre_ay_test_len, pre_ay_val_dataset, pre_ay_val_len
    return pre_ay_train_dataset, pre_ay_test_dataset, pre_ay_train_len, pre_ay_test_len #  ,pre_ay_val_dataset, pre_ay_val_len
    # return pre_ay_train_dataset, pre_ay_val_dataset, pre_ay_train_len, pre_ay_val_len

class NMTCritierion(nn.Module):
    """
    1. Add label smoothing
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

        # print('label', labels[:10])
        # conduct label_smoothing module
        gtruth = labels.view(-1)
        # print('gtruth', gtruth[:10])
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

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 < smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        """
        写法1
        """
        # logprobs = F.log_softmax(x, dim=-1)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)  # 得到交叉熵损失
        # # 注意这里要结合公式来理解，同时留意预测正确的那个类，也有a/K，其中a为平滑因子，K为类别数
        # smooth_loss = -logprobs.mean(dim=1)
        # loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        """
        写法2
        """
        y_hat = torch.softmax(x, dim=1)
        # 这里cross_loss和nll_loss等价
        cross_loss = self.cross_entropy(y_hat, target)
        smooth_loss = -torch.log(y_hat).mean(dim=1)
        # smooth_loss也可以用下面的方法计算,注意loga + logb = log(ab)
        # smooth_loss = -torch.log(torch.prod(y_hat, dim=1)) / y_hat.shape[1]
        loss = self.confidence * cross_loss + self.smoothing * smooth_loss
        return loss.mean()

    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])

if __name__ == '__main__':
    # test_data = [
    #     [109, 0.558, 0.843, 1.198, 1.168, 0.976, 2.298, 0.547, 1.307, 1.078, 1.244, 0],
    #     [181, 0.917, 0.747, 1.528, 1.167, 0.976, 0.265, 0.547, 1.294, 0.616, 0.938, 0],
    #     [323, 0.917, 0.843, 1.528, 0.917, 0.976, 0.265, 0.547, 1.206, 1.078, 1.244, 0],
    #     [609, 0.917, 0.833, 1.345, 1.021, 0.976, 0.265, 0.547, 1.399, 0.336, 0.938, 1],
    #     [868, 0.917, 1.145, 0.606, 1.167, 0.976, 0.586, 0.711, 1.307, 1.2, 0.938, 2]
    # ]
    # # print(stander_data(pd.DataFrame(test_data)))  # , columns=[str(n) for n in range(10)]
    # data_path = r"../data/Anyuan_landslides_10factors_label.csv"
    # a, b, c, d = ay_dataloader(data_path, 400, "logs")
    # for one in a:
    #     print(one["data"].shape)

    # logs = r"logs"
    # print(len(get_data(path, logs)[0][0]))
    ori_data = pd.read_csv(r"I:\mimic_transformer\train\logs2\Transformer_10factors_cell64_ReLU_layer2_drop_ratio0.0_20230423-16-33_epoch2000\features.csv")
    ada_data = pd.read_csv(r"I:\mimic_transformer\source_data_adapted.csv")

    print(stander_data(ori_data))
    print(stander_data(ada_data))