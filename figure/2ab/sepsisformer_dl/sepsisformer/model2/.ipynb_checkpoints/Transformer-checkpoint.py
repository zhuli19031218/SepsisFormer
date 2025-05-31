# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

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
        # print(x.shape)
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
        # print(x.shape)
        x = self.proj_drop(x)
        return x

class FFN(nn.Module):

    def __init__(self, in_features, mlp_ratio=6.0, act_layer=nn.GELU, drop_ration=0.):
        super().__init__()
        hidden = int(in_features*mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(drop_ration)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8,
                 qkv_bias=False,
                 attn_drop_ratio=0.,  #Attention中，线性层后的
                 drop_ratio=0.,       #MLP中的两次dropout
                 drop_path_ratio=0.,  #Attention后的随机深度
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop_ratio=attn_drop_ratio,qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        self.FFN = FFN(in_features=dim, act_layer=act_layer, drop_ration=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.FFN(self.norm2(x)))
        return x
class Transformer(nn.Module):
    def __init__(self, input_dim=8, model_dim=128, depth=8, num_heads=8, # 增加 model_dim 参数
                 qkv_bias=True, attn_drop_ratio=0.1, drop_ratio=0.1, # 增加默认 dropout
                 drop_path_ratio=0.1): # 增加默认 drop path
        super(Transformer, self).__init__()
        self.model_dim = model_dim

        # 1. 嵌入层: input_dim -> model_dim
        self.emmb = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim), # 添加 LayerNorm
            nn.ReLU(),             # 可以加激活函数
            nn.Dropout(drop_ratio)   # 添加 Dropout
            # 不再需要 Conv1d，除非有特殊时序需求
        )

        # 2. Transformer Blocks (使用 model_dim)
        #    确保 Block, Attention, FFN 内部使用 model_dim
        backbone_list = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)] # 随机深度递增
        for i in range(depth):
            backbone_list.append(Block(dim=model_dim, # 使用 model_dim
                                      num_heads=num_heads,
                                      qkv_bias=qkv_bias,
                                      attn_drop_ratio=attn_drop_ratio,
                                      drop_ratio=drop_ratio,
                                      drop_path_ratio=dpr[i])) # 使用递增的 drop path
        self.backbone = nn.Sequential(*backbone_list)

        # 3. 输出头 (从 model_dim -> 2)
        self.norm = nn.LayerNorm(model_dim) # 在 MLP 前加 Norm
        self.MLP = nn.Sequential(
                                  nn.Linear(model_dim, model_dim // 2), # 可以加中间层
                                  nn.ReLU(),
                                  nn.Dropout(drop_ratio),
                                  nn.Linear(model_dim // 2, 2)) # 输出 2 类
    def forward(self,x):
        # x 形状: [Batch, 1, input_dim] = [B, 1, 8]
        # print(f"[Transformer DEBUG] Input shape: {x.shape}")

        x = self.emmb(x) # 输出 [B, 1, model_dim]
        # print(f"[Transformer DEBUG] After emmb shape: {x.shape}")

        # Transformer blocks 期望 [B, Seq, Dim] 或 [Seq, B, Dim]
        # 当前是 [B, 1, model_dim]，符合 [B, Seq, Dim] 格式
        x = self.backbone(x) # 输出 [B, 1, model_dim]
        # print(f"[Transformer DEBUG] After backbone shape: {x.shape}")

        x = self.norm(x) # LayerNorm
        # print(f"[Transformer DEBUG] After norm shape: {x.shape}")

        # MLP 需要作用在特征上，当前形状 [B, 1, model_dim]
        # 取出序列维度（或者平均池化，但这里序列长度是1）
        x = x[:, 0, :] # 取出 [Seq=0] 的数据 -> [B, model_dim]
        # print(f"[Transformer DEBUG] Before MLP shape: {x.shape}")

        x = self.MLP(x) # 输出 [B, 2]
        # print(f"[Transformer DEBUG] After MLP shape: {x.shape}")

        # 不再需要 flatten
        # x = torch.flatten(x, start_dim=1)

        x = torch.nn.Softmax(dim=-1)(x) # 输出 [B, 2]
        # print(f"[Transformer DEBUG] Final output shape: {x.shape}")

        return x
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



if __name__=='__main__':

    input=torch.rand([1, 500, 36])  #torch.float32
    block=Transformer(dim=input.shape[-1],)

    output=block(input)
    print(output.shape)