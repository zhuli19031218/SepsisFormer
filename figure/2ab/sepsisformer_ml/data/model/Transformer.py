import torch
from torch import nn
import math

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.embed= nn.Linear(in_features= d_model, out_features= d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x=torch.tensor(x, dtype=torch.float)
        x=self.embed(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
       
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size,  1, dim]

        B, N, C = x.shape

        # qkv(): -> [batch_size, 1, 3 * dim * num_heads ]
        # reshape: -> [batch_size, 1, 3, num_heads, dim ]
        # permute: -> [3, batch_size, num_heads, 1, dim ]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads,  1, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]       # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, dim,  1]
        # @: multiply -> [batch_size, num_heads,  1,  1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads,  1, dim]
        # transpose: -> [batch_size, 1, num_heads, dim]
        # reshape: -> [batch_size,  1, dim * num_heads]，实现concat
        x = (attn @ v).transpose(1, 2).reshape(B, N, C * self.num_heads)

        # print(x.shape)
        x = self.proj(x)

        x = self.proj_drop(x)

        return x


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x 
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = Attention( d_model, num_heads= d_model)
        # self.norm1 = LayerNorm(d_model=d_model)
        self.norm1 = nn.LayerNorm( d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        # self.norm2 = LayerNorm(d_model=d_model)
        self.norm2 = nn.LayerNorm( d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention

        _x = x
        x = self.attention(x)

        # 2. add and norm
        x = self.dropout1(x)

        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
 
        return x
    
class Transformer(nn.Module):

    def __init__(self, src_pad_idx, enc_voc_size,d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.head1 = nn.Sequential(
                                nn.Linear(d_model, d_model*d_model),
                                nn.LayerNorm(d_model*d_model),
                                  nn.Linear(d_model*d_model, d_model*d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model*d_model, d_model))
        self.head2 = nn.Sequential(
                                nn.Linear(d_model, d_model*d_model),
                                nn.LayerNorm(d_model*d_model),
                                  nn.Linear(d_model*d_model, 2),
                                  nn.GELU(),
                                  nn.Linear(2, 2))
        

    def forward(self, src):
        # src=torch.squeeze(src,1)
        src=self.head1(src)
        enc_src=src
        x=src
        src_mask = self.make_src_mask(src)

        enc_src = self.encoder(src, src_mask)

        enc_src = torch.flatten(enc_src, start_dim=1)

        x=self.head2(enc_src)

        # x=torch.squeeze(x,1)

        x = torch.nn.Softmax(dim=-1)(x)
        return x

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    
if __name__ == '__main__':
    model = Transformer(src_pad_idx=8,
                    d_model=2,
                    enc_voc_size=8,
                    max_len=8,
                    ffn_hidden=64,
                    n_head=8,
                    n_layers=2,
                    drop_prob=0.1,
                    device='cuda:0')
    print(model)
    # input = torch.randint(1, 36, (50,36))
    input = torch.randn(50,1,8)
    output = model(input)
    print(output.shape)