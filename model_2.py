"""
MODEL - 2

TARGETS
* Add more Convolutional Layers
* Reduce number of parameters in the FC layers
* Decrease data sparsity percentage
* Increase Global Receptive Field

RESULTS
* Number of Parameters - 12006
* Train set Accuracy - 99.69%
* Test set Accuracy - 98.82%

ANALYSIS
* Gap between train and test set accuracy is very high in the last few epochs -> Indication of overfitting
* Most learning happens in the FC layers and not in convolutional layers

NEXT STEPS
* Add more Convolutional Layers
* Reduce number of parameters in FC layers

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )# 26X26X16
        self.convblock2 = nn.Sequential(
            nn.Conv2d(4, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )# 24X24X32

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )#10X10X16

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(16, 10, 3, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        # self.convblock5 = nn.Sequential(
        #     nn.Conv2d(10, 10, 1, padding=0),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU()
        # ) # output_size = 8

        self.convblock6 = nn.Sequential(
            nn.Conv2d(10, 10, 10, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        #x = self.convblock5(x)
        x = self.convblock6(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)