import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Sobel Filter
[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
'''


class SobelNet(nn.Module):
    def __init__(self):
        super(SobelNet, self).__init__()
        kernel_1 = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        kernel_2 = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_1 = torch.FloatTensor(kernel_1).unsqueeze(0).unsqueeze(0)
        kernel_2 = torch.FloatTensor(kernel_2).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_1, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_2, requires_grad=False)

    def forward(self, x):
        x1 = F.conv2d(x, self.weight_v, padding=1)
        x2 = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x1, 2) + torch.pow(x2, 2) + 1e-6)
        return x
