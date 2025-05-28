import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv1d(1, 16, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 16, 5])
            nn.Conv1d(16, 32,1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 32, 1])
            
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 32, 1])
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 32, 1])
            
            
            nn.Flatten(),  # torch.Size([128, 32])    (假如上一步的结果为[128, 32, 2]， 那么铺平之后就是[128, 64])
        )
        self.model2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.reshape(-1,1,36)   #结果为[500,1,11]  目的是把二维变为三维数据
        x = self.model1(input)
        x = self.model2(x)
        return x


if __name__ == '__main__':
    model = CNN()
    # print(model)
    input = torch.randn(500, 1, 36)
    # print(input)
    output = model(input)
    # print(output)
    print(output.shape)