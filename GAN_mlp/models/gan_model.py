import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torchvision
from models.generator import G_MLP
from models.discriminator import D_MLP

class GAN_MLP(nn.Module):
    """继承自pytorch的nn神经网络模块"""
    def __init__(self, cfg):
        super().__init__()
        self.name = 'GAN_MLP'
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self._Tensor = torch.cuda.FloatTensor if  torch.cuda.is_available() else torch.Tensor
        self.cfg_G = cfg['G']
        self.cfg_D = cfg['D']
        # self.SEED = cfg['SEED']
        # torch.manual_seed(self.SEED)
        self.lr_G = cfg['lr_G']
        self.lr_D = cfg['lr_D']
        self._batch_size = cfg['batch_size']

        # ----------- initial -------------
        self._init_networks()
        self._init_loss()
        self._init_optimizer()

    def _init_networks(self):
        self._G = G_MLP(input_dim=self.cfg_G['input_dim'], output_dim=self.cfg_G['output_dim'],
                        hidden_layers=self.cfg_G['hidden_layers'], hidden_units=self.cfg_G['hidden_units'])
        self._D = D_MLP(input_dim=self.cfg_D['input_dim'], hidden_layers=self.cfg_D['hidden_layers'],
                        hidden_units=self.cfg_D['hidden_units'])
        # self._G.apply(self.weights_init)        # 使用初始化函数，参考pytorch教程https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        # self._D.apply(self.weights_init)        # 一般在卷积下这种初始化比较好

        self._G = self._G.to(self.device)
        self._D = self._D.to(self.device)

    def _init_loss(self, ):
        self._loss_g_fake = Variable(self._Tensor([0]))     # 在训练生成器的时候，只有一个loss，即由D返回的loss
        self._loss_d_real = Variable(self._Tensor([0]))     # 训练判别器的时候，有两个loss，真样本为一个，由生成器生成的负样本作为另一个
        self._loss_d_fake = Variable(self._Tensor([0]))

    def _init_optimizer(self, ):
        self._D_criterion = nn.BCELoss()
        self._optimizer_D = optim.SGD(self._D.parameters(), lr=self.lr_D, momentum=0.9)
        self._optimizer_G = optim.SGD(self._G.parameters(), lr=self.lr_G, momentum=0.9)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        with torch.no_grad():
            out = self._G(z).detach().cpu()
        return out

    def optimize_G(self, z=None):
        '''When optiimze G, the input is a vector z, which is randomly sampled from Gaussian distribution'''
        self._optimizer_G.zero_grad()

        # First, we randomly generated fake samples
        # Latent vector z: 即每次先产生一个z向量，为128维，指定送到self.device
        if z is None:
            z = torch.randn(self._batch_size, self.cfg_G['input_dim'], device=self.device)
        fake_sample = self._G(z)  # Generate fake samples
        fake_score = self._D(fake_sample)
        self._loss_g_fake = self._compute_loss_D(fake_score, is_real=True)
        # 为什么这里使用True呢？因为G的目标是最大化生成样本在D的输出，但是由于是梯度下降优化过程，所以使用real的标签，就可以让D给出对
        # 真实样本的分数，这分数越高，
        self._loss_g_fake.backward()
        self._optimizer_G.step()
        return self._loss_g_fake.item(), fake_sample

    def optimize_D(self, x):
        '''When optimize D, the input sample is real sample and the generated images by G优化D的时候，送入的样本为真，同时使用G生成负样本'''
        self._optimizer_D.zero_grad()

        # we input the real sample to D
        real_score = self._D(x)
        # Second, we randomly generated fake samples
        # Latent vector z: 即每次先产生一个z向量，为128维，指定送到self.device
        z = torch.randn(self._batch_size, self.cfg_G['input_dim'], device=self.device)
        fake_sample = self._G(z)                #   Generate fake samples
        fake_score = self._D(fake_sample)
        # Compute loss
        self._loss_d_real = self._compute_loss_D(real_score, is_real=True)
        self._loss_d_fake = self._compute_loss_D(fake_score, is_real=False)
        loss = self._loss_d_real + self._loss_d_fake

        loss.backward()
        self._optimizer_D.step()
        return loss.item(), fake_sample

    def _compute_loss_D(self, output, is_real):
        label = torch.full((output.size(0), ), is_real, device=self.device)
        return self._D_criterion(output, label)

