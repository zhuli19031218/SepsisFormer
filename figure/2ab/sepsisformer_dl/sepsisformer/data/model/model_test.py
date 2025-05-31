import torch
import torch.nn as nn
import torch.nn.functional as F
from AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Autoformer_EncDec import Encoder,  EncoderLayer,  my_Layernorm, series_decomp
import math
import numpy as np

class Mlp(nn.Module):

    def __init__(self, in_features, mlp_ratio=3.0, act_layer=nn.GELU, drop_ration=0.1):
        super().__init__()
        hidden=int(in_features*mlp_ratio)
        self.fc1 = nn.Linear(in_features,hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, in_features)
        self.fc3 = nn.Linear(in_features,2)
        self.drop = nn.Dropout(drop_ration)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x.shape1',x.shape)
        x = self.fc1(x)
        # print('x.shape2',x.shape)
        x = self.drop(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        # x = self.sigmoid(x)
        x = self.softmax(x)
        return x


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, factor=1,input_size=36,moving_avg=25,device='cuda:0',
                 seq_len=36,label_len=1,pred_len=2,d_model=36, n_heads=8,
                 dropout=0.1,d_ff=None,activation='gelu',
                 e_layers=8,#循环几次encoder
                 output_attention=False):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=input_size * d_model)
        #利用历史时间序列的时间戳长度，编码器输入的时间维度
        self.seq_len = seq_len
        #解码器输入的历史时间序列的时间戳长度
        self.label_len = label_len
        self.pred_len = pred_len
        self.input_size = input_size
        self.output_attention =output_attention
        self.dim = d_model
        self.device = device
        self.mlp = Mlp(in_features=d_model * 36)
        
        # Encoder,采用的是多编码层堆叠
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        #这里的第一个False表明是否使用mask机制。
                        AutoCorrelation(False, factor, 
                                        attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    #编码过程中的特征维度设置
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    #激活函数
                    activation=activation
                ) for l in range(e_layers)
            ],
            #时间序列通常采用Layernorm而不适用BN层
            norm_layer=my_Layernorm(d_model)
        )


        self.fc3 = nn.Linear(input_size*d_model,2)
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax(dim=-1)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self,input,
                enc_self_mask=None):
        
        x = input.unsqueeze(dim=1)
        # print(x.shape)
        x = self.linear(x).to(self.device)
        enc_out = x.reshape(-1, self.input_size, self.dim)
        # print(enc_out.shape)#[500, 11, 11]
        
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print(enc_out.shape)
        flatten = torch.flatten(enc_out, start_dim=1, end_dim=-1)
        # print("flatten",flatten.shape)
        # output=self.fc3(flatten)
        # output=self.drop(output)
        # output=self.softmax(output)
        output = self.mlp(flatten)
        return output

if __name__ == '__main__':
    model = Model(device='cpu')
    input = torch.randn(500, 1, 36)
    output = model(input)
    print(output.shape)
       