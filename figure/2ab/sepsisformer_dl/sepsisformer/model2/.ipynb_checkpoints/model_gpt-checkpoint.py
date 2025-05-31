import numpy as np
from torch import nn,optim
from torch.nn.functional import cross_entropy,softmax, relu
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import torch

import os
import pickle

# MultiHeadAttention 类保持不变 (请确保它在您的环境中可用)
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
        # 确保 heads_dim > 0
        assert hid_size % n_heads == 0, "hid_size 必须能被 n_heads 整除"
        self.heads_dim=hid_size//n_heads
        self.scale=torch.sqrt(torch.FloatTensor([self.heads_dim])).to(device)
        
        self.dropout=nn.Dropout(dropout)
        self.device=device
    
    def forward(self,q,k,v,masked=None):
        #q[batch seq_len hid_size]
        #k[batch seq_len hid_size]
        #v[batch seq_len  hid_size]
        
        #首先经历三个线性变化得到q,v,k向量
        # 注意: .to(self.device) 应该在创建张量时或模型初始化时完成，
        #       线性层会自动处理输入张量的device，无需手动转换q,k,v
        q_lin=self.Q(q)
        k_lin=self.K(k)
        v_lin=self.V(v)
        
        #q[batch seq_len hid_size]
        #k[batch seq_len hid_size]
        #v[batch seq_len  hid_size]     
        
        batch=q.shape[0]
        # print("q",q.shape)
        #由于是多头自注意力，我们将维度hid_size分成n_heads份
        #每一个多头我们希望其关注不同侧重点
        q_reshaped=q_lin.reshape(batch,-1,self.n_heads,self.heads_dim)
        #q[batch seq_len n_heads heads_dim]
        q_permuted=q_reshaped.permute(0,2,1,3)
        #q[batch n_heads seq_len heads_dim]       
        k_permuted=k_lin.reshape(batch,-1,self.n_heads,self.heads_dim).permute(0,2,1,3)
        v_permuted=v_lin.reshape(batch,-1,self.n_heads,self.heads_dim).permute(0,2,1,3)
        
        #计算注意力权重
        #q[batch n_heads seq_len1 heads_dim]   
        #k[batch n_heads seq_len heads_dim]   
        #v[batch n_heads seq_len heads_dim] 
        
        # .to(self.device) 应在 self.scale 初始化时完成
        energy=torch.matmul(q_permuted, k_permuted.permute(0,1,3,2)) / self.scale
        #energy[batch n_head seq_len1 seq_len]
        
        #将energy通进行mask忽视pad
        if masked is not None:
            # 确保 masked 张量在正确的 device 上
            energy=energy.masked_fill(masked.to(self.device)==0,-1e10)
        
        attention=torch.softmax(energy,dim=-1)
        #attention[batch n_head seq_len1 seq_len]
        
        #对权重与值向量加权求和得到上下文向量
        # dropout 应用在 attention 权重上
        context=torch.matmul(self.dropout(attention),v_permuted)
        #context[batch n_head seq_len1 heads_dim]
        
        #拼接各个头并进行维度变化输出
        # contiguous() 确保内存连续，有时在 reshape 前需要
        context=context.permute(0,2,1,3).contiguous().reshape(batch,-1,self.hid_size)
        #context[batch seq_len hid_size]
        output=self.fc(context)
        return output,attention

# 修改 TransformerDecoder 以接受 ff_expansion_factor
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_expansion_factor, dropout, device='cuda'):
        super().__init__()
        self.device = device # 保存device

        # MultiHeadAttention 初始化时传入 device 和 dropout
        self.multihead_attn = MultiHeadAttention(hid_size=hidden_dim, n_heads=num_heads, device=device, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        # FFN 使用 ff_expansion_factor 来决定中间层维度
        ff_intermediate_dim = hidden_dim * ff_expansion_factor
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_intermediate_dim),
            nn.ReLU(), # 或者使用 nn.GELU()，更常见于Transformer
            nn.Dropout(dropout),
            nn.Linear(ff_intermediate_dim, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    # forward 方法需要接收 mask
    def forward(self, x, masked=None):
        # Multi-head self-attention
        # 将 mask 传递给 MultiHeadAttention
        attn_out, _ = self.multihead_attn(x, x, x, masked=masked)
        attn_out = self.dropout1(attn_out)
        x = self.layer_norm1(x + attn_out) # 残差连接和LayerNorm

        # Feedforward neural network
        ff_out = self.ff(x)
        ff_out = self.dropout2(ff_out)
        x = self.layer_norm2(x + ff_out) # 残差连接和LayerNorm

        return x

# 修改后的 GPT 模型
class GPT(nn.Module):
    def __init__(self, num_tokens=6, hidden_dim=11, num_heads=11, num_layers=6,
                 ff_expansion_factor=4, dropout=0.1, max_seq_len=512, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        # 输入嵌入层 (如果输入是token ID，则需要)
        # self.token_embedding = nn.Embedding(num_tokens, hidden_dim)
        # 假设输入已经是 [batch, seq_len, hidden_dim] 的嵌入向量

        # 位置编码层
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout) # Dropout for embeddings

        # Transformer Decoder 层列表
        self.decoders = nn.ModuleList([
            TransformerDecoder(hidden_dim, num_heads, ff_expansion_factor, dropout, device=device)
            for _ in range(num_layers)
        ])

        # 简化的 Head 层，类似 LSTM 的输出层结构
        # 从 hidden_dim 映射到最终的输出维度 (这里是 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), # LayerNorm 在最后线性层之前
            nn.Linear(hidden_dim, 2)
        )
        # Softmax 将在 forward 的最后应用

        # 将模型的所有参数移动到指定设备
        self.to(device)

    def forward(self, x, masked=None):
        # x shape: [batch, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.shape
        assert seq_len <= self.pos_embedding.num_embeddings, \
            f"输入序列长度 {seq_len} 超过了最大允许长度 {self.pos_embedding.num_embeddings}"

        # 1. 添加位置编码
        # 创建位置索引 [0, 1, ..., seq_len-1]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0) # shape: [1, seq_len]
        pos_emb = self.pos_embedding(pos) # shape: [1, seq_len, hidden_dim]
        # 将位置编码加到输入 x 上 (使用广播机制)
        x = x + pos_emb
        x = self.dropout(x)

        # 2. 通过 Transformer Decoder 层
        for decoder in self.decoders:
            # 将 mask 传递给每个 decoder 层
            x = decoder(x, masked=masked)
        # x shape: [batch, seq_len, hidden_dim]


        x_last = x[:, -1, :] # 取最后一个时间步的输出, shape: [batch, hidden_dim]
        output = self.head(x_last) # shape: [batch, 2]


        output = torch.softmax(output, dim=-1) # 在最后一个维度（类别维度）上应用Softmax

        return output
