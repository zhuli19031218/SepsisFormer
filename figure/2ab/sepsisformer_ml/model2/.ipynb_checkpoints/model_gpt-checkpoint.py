
import numpy as np
from torch import nn,optim
from torch.nn.functional import cross_entropy,softmax, relu
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import torch

import os
import pickle

class MultiHeadAttention(nn.Module):
    def __init__(self,hid_size,n_heads,device='cuda',dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        
        #三个线性层模板
        self.Q=nn.Linear(hid_size,hid_size)
        self.K=nn.Linear(hid_size,hid_size)
        self.V=nn.Linear(hid_size,hid_size)
        
        #多头拼接层
        self.fc=nn.Linear(hid_size,hid_size)
        
        #各种参数记录
        self.hid_size=hid_size
        self.n_heads=n_heads
        self.heads_dim=hid_size//n_heads
        self.scale=torch.sqrt(torch.FloatTensor([self.heads_dim])).to(device)
        
        self.dropout=nn.Dropout(dropout)
        self.device=device
    
    def forward(self,q,k,v,masked=None):
        #q[batch seq_len hid_size]
        #k[batch seq_len hid_size]
        #v[batch seq_len  hid_size]
        
        #首先经历三个线性变化得到q,v,k向量
        q=self.Q(q)
        k=self.K(k)
        v=self.V(v)
        
        #q[batch seq_len hid_size]
        #k[batch seq_len hid_size]
        #v[batch seq_len  hid_size]     
        
        batch=q.shape[0]
        # print("q",q.shape)
        #由于是多头自注意力，我们将维度hid_size分成n_heads份
        #每一个多头我们希望其关注不同侧重点
        q=q.reshape(batch,-1,self.n_heads,self.heads_dim)
        #q[batch seq_len n_heads heads_dim]
        q=q.permute(0,2,1,3)
        #q[batch n_heads seq_len heads_dim]       
        k=k.reshape(batch,-1,self.n_heads,self.heads_dim).permute(0,2,1,3)
        v=v.reshape(batch,-1,self.n_heads,self.heads_dim).permute(0,2,1,3)
        
        #计算注意力权重
        #q[batch n_heads seq_len1 heads_dim]   
        #k[batch n_heads seq_len heads_dim]   
        #v[batch n_heads seq_len heads_dim] 
        
        energy=torch.matmul(q,k.permute(0,1,3,2))/self.scale
        #energy[batch n_head seq_len1 seq_len]
        
        #将energy通进行mask忽视pad
        if masked is not None:
            energy=energy.masked_fill(masked==0,-1e10)
        
        attention=torch.softmax(energy,dim=-1)
        #attention[batch n_head seq_len1 seq_len]
        
        #对权重与值向量加权求和得到上下文向量
        context=torch.matmul(self.dropout(attention),v)
        #context[batch n_head seq_len1 heads_dim]
        
        #拼接各个头并进行维度变化输出
        context=context.permute(0,2,1,3).reshape(batch,-1,self.hid_size)
        #context[batch seq_len hid_size]
        output=self.fc(context)
        return output,attention

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout):
        super().__init__()

        self.multihead_attn = MultiHeadAttention(hid_size=hidden_dim, n_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Multi-head self-attention
        # print(x.shape)
        attn_out, _ = self.multihead_attn(x, x, x)
        attn_out = self.dropout1(attn_out)
        x = self.layer_norm1(x + attn_out)

        # Feedforward neural network
        ff_out = self.ff(x)
        ff_out = self.dropout2(ff_out)
        x = self.layer_norm2(x + ff_out)

        return x



class GPT(nn.Module):
    def __init__(self, num_tokens=6, hidden_dim=11, num_heads=11, num_layers=6,  dropout=0.1):
        super().__init__()

        self.decoders = nn.ModuleList([
            TransformerDecoder(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
                                
                                  nn.LayerNorm(hidden_dim),
                                  nn.Linear(hidden_dim, 2),
                                  nn.GELU(),
                                  nn.Linear(2, 2))
        self.output_layer = nn.Linear(hidden_dim, num_tokens)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # print(x.shape)
 
        for decoder in self.decoders:
            x = decoder(x)
        x=self.head(x)
        # print(x.shape)
        # Output layer
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = torch.nn.Softmax(dim=-1)(x)

        return x


if __name__=='__main__':
    

    input=torch.rand([64,1, 11])  #torch.float32
    model = GPT()

    output=model(input)
    print('output',output.shape)