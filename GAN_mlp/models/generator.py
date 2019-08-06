import torch
import torch.nn as nn

class G_MLP(nn.Module):
    """继承自pytorch的nn神经网络模块"""
    def __init__(self, input_dim=128, output_dim=28*28, hidden_layers=2, hidden_units=256):
        super().__init__()
        self.name = 'G_MLP'

        self.input_layer = nn.Linear(input_dim, hidden_units)
        self.lrelu = nn.LeakyReLU()
        # 为了使用hidden_layer这个参数，我们将用到nn.Sequential()函数，这类似于keras的sequential
        layers = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units * 2))
            layers.append(nn.BatchNorm1d(hidden_units * 2))     # 使用BatchNorm
            layers.append(nn.LeakyReLU())
            hidden_units *= 2
        self.hidden = nn.Sequential(*layers)    # 模块化

        self.output = nn.Sequential(
            nn.Linear(hidden_units, output_dim),
            nn.Tanh(),
        )
        # 最后一层加Tanh激活函数输出，或者你也可以使用sigmoid，但是数据预处理就不能normalize

    def forward(self, z):
        '''input a latent vector z, output a 1 x output_dim vector, which defines a generated image'''
        out = self.lrelu(self.input_layer(z))
        out = self.hidden(out)
        out = self.output(out)

        return out