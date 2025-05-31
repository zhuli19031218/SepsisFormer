import torch.nn as nn
import torch


class Lstm(nn.Module):
    def __init__(self,
                 factors,  # 输入token的dim
                 batch_size,
                 device,
                 drop_ratio,

                 ):
        super(Lstm, self).__init__()
        self.factors = factors
        self.batch_size = batch_size
        self.device = device


        self.backbone1 = nn.LSTM(input_size=factors, hidden_size=24 * factors, num_layers=1)
        self.backbone2 = nn.LSTM(input_size= 24*factors, hidden_size=24 * factors, num_layers=1)
        self.backbone3 = nn.LSTM(input_size=24 *factors, hidden_size=4 * factors, num_layers=1)

        # self.emmb= nn.Sequential(
        #     nn.Linear(factors, factors),
        #     # nn.ReLU(),
        #     # nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1),
        #     #nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        #     )

        self.head = nn.Sequential(
            nn.Linear(in_features=4 * factors, out_features=2 * factors),
            nn.Dropout(drop_ratio),
            nn.Linear(in_features=2 * factors, out_features=2),
            nn.Dropout(drop_ratio),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        # print('input', input.shape)
        #print('self.batch_siz',self.batch_siz)
        h0 = torch.randn(1, input.shape[1], 24*self.factors).to(self.device)  # (num_layers,batch,output_size)
        c0 = torch.randn(1, input.shape[1], 24*self.factors).to(self.device)  # (num_layers,batch,output_size)
        #output, _ = self.backbone1(input, (h0, c0))  # 可计算出output和(hn, cn)
        #print('1', output.shape)
        output, _ = self.backbone1(input, (h0, c0))
        output, _ = self.backbone2(output)
        # print('2', output.shape)
        # output = self.emmb(input)

        output, _ = self.backbone3(output)
        # print('3', output.shape)
        output = self.head(output)
        # print('head', output.shape)
        output = torch.squeeze(output)
        # print('suqeeze', output.shape)
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Lstm(factors=8, batch_size=200, device='cuda', drop_ratio=0.).to(device)
    input = torch.randn(1, 200, 8).to(device)
    output = model(input)
    print(output.shape)
    # input = torch.randn(50, 1, 21)



# import torch.nn as nn
# import torch


# class Lstm(nn.Module):
#     def __init__(self,
#                  factors,  # 输入token的dim
#                  hidden_size,
#                  batch_size,
#                  device,
#                  drop_ratio,
#                  ):
#         super(Lstm, self).__init__()
#         self.factors = factors
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
#         self.device = device

#         self.backbone1 = nn.LSTM(input_size=factors, hidden_size= hidden_size, num_layers=1)
#         # self.backbone1 = nn.LSTM(input_size=factors, hidden_size=256, num_layers=2)
#         self.backbone2 = nn.LSTM(input_size=hidden_size, hidden_size= hidden_size, num_layers=1)
#         self.backbone3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)

#         self.head = nn.Sequential(
#             nn.Linear(in_features=hidden_size, out_features=hidden_size*2),
#             nn.Dropout(drop_ratio),
#             nn.ReLU(),
#             nn.Linear(in_features=hidden_size*2, out_features=2),
#             nn.Dropout(drop_ratio),
#             nn.ReLU(),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, input):
#         input=input.permute(1,0,2)
#         # print(input.shape)
#         # h0 = torch.randn(1, input.shape[1], self.factors).to(self.device)  # (num_layers,batch,output_size)
#         # c0 = torch.randn(1, input.shape[1], self.factors).to(self.device)  # (num_layers,batch,output_size)
#         h0 = torch.randn(1, input.shape[1], self.hidden_size).to(self.device)  # (num_layers,batch,output_size)
#         c0 = torch.randn(1, input.shape[1], self.hidden_size).to(self.device)
#         output, _ = self.backbone1(input, (h0, c0))  # 可计算出output和(hn, cn)
#         # print('1', output.shape)
#         output1 = output
#         output, _ = self.backbone2(output1, _)
#         # print('2', output.shape)
#         output2 = output 
#         output, _ = self.backbone3(output2, _)
#         # print('3', output.shape)
#         output = output 
#         # print("4",output.shape)
#         output = self.head(output)
#         # print('head', output.shape)
#         output = torch.squeeze(output)
#         # print('suqeeze', output.shape)
#         return output


# if __name__ == '__main__':
#     model = Lstm(factors=8, hidden_size=64, batch_size=100, device='cpu', drop_ratio=0.)
#     input = torch.randn(100,1, 8)
#     output = model(input)
#     print(output.shape)
#     # input = torch.randn(50, 1, 21)








