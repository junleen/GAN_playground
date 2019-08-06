import torch
import torch.nn as nn

class D_MLP(nn.Module):
    """继承自pytorch的nn神经网络模块"""
    def __init__(self, input_dim=28*28, hidden_layers=2, hidden_units=512):
        super().__init__()
        self.name = 'D_MLP'

        self.input_layer = nn.Linear(input_dim, hidden_units)
        self.lrelu = nn.ReLU()
        # 为了使用hidden_layer这个参数，我们将用到nn.Sequential()函数，这类似于keras的sequential
        layers = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units // 2))
            layers.append(nn.BatchNorm1d(hidden_units // 2))     # 使用BatchNorm
            layers.append(nn.ReLU())
            hidden_units = hidden_units // 2
        self.hidden = nn.Sequential(*layers)    # 模块化

        self.output = nn.Sequential(
            nn.Linear(hidden_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''input a image x, output a single scalar, which means real/fake score'''
        out = self.lrelu(self.input_layer(x))
        out = self.hidden(out)
        out = self.output(out)
        out = out.view(-1)
        return out