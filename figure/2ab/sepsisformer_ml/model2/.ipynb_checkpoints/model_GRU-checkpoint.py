import torch.nn as nn
import torch
# from comparison_utilis_data import *
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

class GRU(nn.Module):
    def __init__(self,
                 factors,  # 输入token的dim
                 batch_size,
                 device,
                 drop_ratio,
                 num_layers
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=factors, hidden_size=16*factors, num_layers=self.num_layers)
        # utilize the GRU model in torch.nn
        self.head1 = nn.Sequential(
            nn.Linear(in_features=factors, out_features=256),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(in_features=256, out_features=8),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=16* factors, out_features=128),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(in_features=128, out_features=2)
        )


    def forward(self, _x):
        # _x = _x.transpose(0, 1)
        # print(_x.shape)
        self.gru.flatten_parameters()
        x = self.head1(_x)
        x, _ = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
        # print(x.shape)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = self.head(x)
        x = torch.squeeze(x)
        return x





if __name__ == '__main__':
    model = GRU(factors=8, batch_size=200, device='cpu', drop_ratio=0.2,num_layers=2)  # ,epoch4000
    input = torch.randn(1, 200, 8)
    output = model(input)
    print(output.shape)
