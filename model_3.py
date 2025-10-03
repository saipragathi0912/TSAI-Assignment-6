"""
MODEL - 3

TARGETS
* Experiment with different learning rates, optimizers and schedulers to boost training

RESULTS
* Number of Parameters - 7598
* Train set Accuracy - 99.55%
* Test set Accuracy - 99.09%

ANALYSIS
* Using AdamW optimizer and OneCycleLR worked best to boost the learning process during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 11, 3, padding=0),
            nn.BatchNorm2d(11),
            nn.ReLU()
        )# 26X26X16 RF - 3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(11, 22, 3),
            nn.BatchNorm2d(22),
            nn.ReLU()
        )# 24X24X32 RF - 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(22, 22, 1),
            nn.BatchNorm2d(22),
            nn.ReLU()
        )#10X10X16 

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(22, 22, 3),
            nn.BatchNorm2d(22),
            nn.ReLU() 
        ) # RF - 7
        self.convblock5 = nn.Sequential(
            nn.Conv2d(22, 10, 1, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 8 RF - 9
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1 RF - 15
        # self.convblock6 = nn.Sequential(
        #     nn.Conv2d(10, 10, 10, padding=0, bias=False),
        # )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.gap(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)