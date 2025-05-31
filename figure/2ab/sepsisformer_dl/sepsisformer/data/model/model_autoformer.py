import torch
import torch.nn as nn
import torch.nn.functional as F

from AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Autoformer_EncDec import Encoder,  EncoderLayer,  my_Layernorm, series_decomp
import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=36.0, act_layer=nn.GELU, drop_ration=0.1):
        super().__init__()
        hidden=int(in_features*mlp_ratio)
        self.fc1 = nn.Linear(in_features,hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, in_features)
        self.fc3 = nn.Linear(in_features,2)
        self.drop = nn.Dropout(drop_ration)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.act(x)
        
        x = self.fc2(x)
        x = self.drop(x)
        x = self.act(x)
        
        x = self.fc3(x)
        x = self.drop(x)
        x = self.act(x)
        
        x = self.softmax(x)
        return x


class AutoFormer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, factor=5,input_size=36,moving_avg=25,device='cuda:0',
                 seq_len=36,label_len=1,pred_len=2,d_model=36, n_heads=8,
                 dropout=0.05,d_ff=None,activation='gelu',
                 e_layers=8,#循环几次encoder
                 output_attention=False):
        super(AutoFormer, self).__init__()
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
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.input_size*seq_len,input_size)
        self.mlp = Mlp(in_features=d_model)

        
    def forward(self,input,
                enc_self_mask=None):   
        x = input.unsqueeze(dim=1)
        x = self.linear(x).to(self.device)
        
        enc_out = x.reshape(-1, self.input_size, self.dim)       
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
       
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
       
        flatten=output
        # flatten = torch.flatten(enc_out, start_dim=1, end_dim=-1)
        # print(flatten.shape)
        output = self.mlp(flatten)
        return output

if __name__ == '__main__':
    model = AutoFormer(device='cpu')
    print(model)
    input = torch.randn(500, 1, 36)
    output = model(input)
    print(output.shape)


# Model(
#   (linear): Linear(in_features=36, out_features=1296, bias=True)
#   (mlp): Mlp(
#     (fc1): Linear(in_features=1296, out_features=3888, bias=True)
#     (act): GELU(approximate='none')
#     (fc2): Linear(in_features=3888, out_features=1296, bias=True)
#     (fc3): Linear(in_features=1296, out_features=2, bias=True)
#     (drop): Dropout(p=0.2, inplace=False)
#     (softmax): Softmax(dim=-1)
#     (sigmoid): Sigmoid()
#   )
#   (encoder): Encoder(
#     (attn_layers): ModuleList(
#       (0-4): 5 x EncoderLayer(
#         (attention): AutoCorrelationLayer(
#           (inner_correlation): AutoCorrelation(
#             (dropout): Dropout(p=0.05, inplace=False)
#           )
#           (query_projection): Linear(in_features=36, out_features=36, bias=True)
#           (key_projection): Linear(in_features=36, out_features=36, bias=True)
#           (value_projection): Linear(in_features=36, out_features=36, bias=True)
#           (out_projection): Linear(in_features=36, out_features=36, bias=True)
#         )
#         (conv1): Conv1d(36, 144, kernel_size=(1,), stride=(1,), bias=False)
#         (conv2): Conv1d(144, 36, kernel_size=(1,), stride=(1,), bias=False)
#         (decomp1): series_decomp(
#           (moving_avg): moving_avg(
#             (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
#           )
#         )
#         (decomp2): series_decomp(
#           (moving_avg): moving_avg(
#             (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
#           )
#         )
#         (dropout): Dropout(p=0.05, inplace=False)
#       )
#     )
#     (norm): my_Layernorm(
#       (layernorm): LayerNorm((36,), eps=1e-05, elementwise_affine=True)
#     )
#   )
#   (fc3): Linear(in_features=1296, out_features=2, bias=True)
#   (drop): Dropout(p=0.5, inplace=False)
#   (softmax): Softmax(dim=-1)
# ) 