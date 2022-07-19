# this code is from below link
# https://gitlab.com/qbeer/loss-landscape/-/blob/main/loss_landscape/models.py

# import torch
# model_ids = ['mobilenet_v2', 'vgg11', 'vgg11_bn', 'alexnet', 'vgg16', 'vgg16_bn', 'resnet18',
#              'densenet161', 'inception_v3',
#              'googlenet', 'resnext50_32x4d', 'mnasnet1_0',
#              'resnet50']
# def load_model(model_identifier):
#     return torch.hub.load('pytorch/vision:v0.6.0', model_identifier, pretrained=True, verbose=False).eval()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AutoEncoder(nn.Module) :
    def __init__(self) :
        super().__init__()
        # self.act = nn.ReLU()
        self.act = nn.GELU()
        # self.act = nn.LeakyReLU()
        
        self.dim = 30
        self.hidden1 = 64
        self.hidden2 = 128
        self.drop_rate = 0.
        
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.dim),
            nn.Linear(self.dim,self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            self.act,
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.hidden1,self.hidden2),
            nn.BatchNorm1d(self.hidden2),
            self.act,
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden2,self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            self.act,
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.hidden1,self.dim),
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)          
        
    def forward(self, x) :
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
class TiedAE(nn.Module) :
    def __init__(self) :
        super().__init__()
        # self.act = nn.ReLU()
        self.act = nn.GELU()
        # self.act = nn.LeakyReLU()
        
        self.dim = 30
        self.hidden1 = 64
        self.hidden2 = 128
        self.drop_rate = 0.
        
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.dim),
            nn.Linear(self.dim,self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            self.act,
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.hidden1,self.hidden2),
            nn.BatchNorm1d(self.hidden2),
            self.act,
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden2,self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            self.act,
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.hidden1,self.dim),
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)          
        
    def forward(self, x) :
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x    
    
class StackedAE(nn.Module) :
    def __init__(self) :
        super().__init__()
        # self.act = nn.ReLU()
        self.act = nn.GELU()
        # self.act = nn.LeakyReLU()
        
        self.dim = 30
        self.hidden1 = 64
        self.hidden2 = 128
        self.drop_rate = 0.
        
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.dim),
            nn.Linear(self.dim,self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            self.act,
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.hidden1,self.hidden2),
            nn.BatchNorm1d(self.hidden2),
            self.act,
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden2,self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            self.act,
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.hidden1,self.dim),
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)          
        
    def forward(self, x) :
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x     
