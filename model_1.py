"""
MODEL - 1

TARGETS
* Get the skeleton code ready
* Model should be able to learn something 
* Do not care about the number of paramters


RESULTS
* Number of Parameters ~ 37M
* Train set Accuracy - 100%
* Test set Accuracy - 98.89%

ANALYSIS
* Gap between train and test set accuracy is very high in the last few epochs -> Indication of overfitting
* Most learning happens in the FC layers and not in convolutional layers
* On visualization and analysis, data sparsity in the channels are too high. Maybe we don't need that many channels

NEXT STEPS
* Add more Convolutional Layers
* Reduce number of parameters in FC layers
* Reduce gap between train and test accuracies
* Decrease the number of channels to handle data sparsity

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=0),
            nn.ReLU()
        )# 26X26X32
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )# 24X24X64

        self.fc1 = nn.Linear(24*24*64, 1028)
        self.fc2 = nn.Linear(1028, 10)
        #self.fc2 = nn.Linear(10*10*1, 10)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x.view(-1, 24*24*64)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)