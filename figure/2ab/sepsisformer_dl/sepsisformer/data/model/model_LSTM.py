import torch.nn as nn
import torch


class Lstm(nn.Module):
    def __init__(self,
                 factors,  # 输入token的dim
                 hidden_size,
                 batch_size,
                 device,
                 drop_ratio,
                 ):
        super(Lstm, self).__init__()
        self.factors = factors
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device

        self.backbone1 = nn.LSTM(input_size=factors, hidden_size= 128, num_layers=1)
        # self.backbone1 = nn.LSTM(input_size=factors, hidden_size=256, num_layers=2)
        self.backbone2 = nn.LSTM(input_size=128, hidden_size= 128, num_layers=1)
        self.backbone3 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1)

        self.head = nn.Sequential(
            nn.Linear(in_features=128, out_features=hidden_size*2),
            nn.Dropout(drop_ratio),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size*2, out_features=2),
            nn.Dropout(drop_ratio),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        input=input.permute(1,0,2)
        # print(input.shape)
        # h0 = torch.randn(1, input.shape[1], self.factors).to(self.device)  # (num_layers,batch,output_size)
        # c0 = torch.randn(1, input.shape[1], self.factors).to(self.device)  # (num_layers,batch,output_size)
        h0 = torch.randn(1, input.shape[1], 128).to(self.device)  # (num_layers,batch,output_size)
        c0 = torch.randn(1, input.shape[1], 128).to(self.device)
        output, _ = self.backbone1(input, (h0, c0))  # 可计算出output和(hn, cn)
        # print('1', output.shape)
        output1 = output
        output, _ = self.backbone2(output1, _)
        # print('2', output.shape)
        output2 = output 
        output, _ = self.backbone3(output2, _)
        # print('3', output.shape)
        output = output 
        # print("4",output.shape)
        output = self.head(output)
        # print('head', output.shape)
        output = torch.squeeze(output)
        # print('suqeeze', output.shape)
        return output


if __name__ == '__main__':
    model = Lstm(factors=8, hidden_size=128, batch_size=100, device='cpu', drop_ratio=0.)
    input = torch.randn(100,1, 8)
    output = model(input)
    print(output.shape)
    # input = torch.randn(50, 1, 21)








