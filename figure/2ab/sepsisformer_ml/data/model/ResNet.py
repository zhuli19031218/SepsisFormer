import torch
from torchsummary import summary
import torch.nn as nn
# 残差块
class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual


class ResNet(torch.nn.Module):
    def __init__(self,in_channels=2,classes=2):
        super(ResNet, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=1,stride=1,padding=1),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            
            Bottlrneck(256,128,512, True),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            
            Bottlrneck(1024,512,2048, True),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,classes)
        )

    def forward(self,x):
        x = self.features(x)

        x = x.view(-1,2048)

        x = self.classifer(x)
        return x

if __name__ == '__main__':
    # x = torch.randn(size=(1,1,224))
    x = torch.randn(size=(2667,1,36))
    # model = Bottlrneck(64,64,256,True)
    model = ResNet(in_channels=1)

    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    print(model)




# # 残差块
# class Bottlrneck(torch.nn.Module):
#     def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
#         super(Bottlrneck, self).__init__()
#         self.stride = 1
#         if downsample == True:
#             self.stride = 2

#         self.layer = torch.nn.Sequential(
#             torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
#             torch.nn.BatchNorm1d(Med_channel),
#             torch.nn.ReLU(),
#             torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
#             torch.nn.BatchNorm1d(Med_channel),
#             torch.nn.ReLU(),
#             torch.nn.Conv1d(Med_channel, Out_channel, 1),
#             torch.nn.BatchNorm1d(Out_channel),
#             torch.nn.ReLU(),
#         )

#         if In_channel != Out_channel:
#             self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
#         else:
#             self.res_layer = None

#     def forward(self,x):
#         if self.res_layer is not None:
#             residual = self.res_layer(x)
#         else:
#             residual = x
#         return self.layer(x)+residual


# class ResNet(torch.nn.Module):
    def __init__(self,in_channels=2,classes=2):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=3,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            
            Bottlrneck(256,128,512, True),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            
            Bottlrneck(1024,512,2048, True),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,classes)
        )

    def forward(self,x):
        print("1",x.shape)
        x = self.features(x)
        print("2",x.shape)
        x = x.view(-1,2048)
        print("3",x.shape)
        x = self.classifer(x)
        return x