
from __future__ import print_function
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms

import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings(action='ignore')

from datetime import datetime
import wandb

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

from dataset import CDataset
from models import AutoEncoder

parser = argparse.ArgumentParser(description='Credit Card Fraud Detection')
parser.add_argument('--data_path', metavar='DIR', default='./dataset/',
                    help='path to dataset (default: ./dataset/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ae_basic',
                    help='model architecture: (default: ae_basic)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='W', help='optimizer (default: adam)')


def seed_everything(seed) :
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def build_dataset(config) :
    train_df = pd.read_csv(config.data_path + 'train.csv')
    val_df = pd.read_csv(config.data_path + 'val.csv')
    test_df = pd.read_csv(config.data_path + 'test.csv')
    train_df = train_df.drop(columns=['ID'])
    val_df = val_df.drop(columns=['ID'])  
    test_df = test_df.drop(columns=['ID'])  
    print(train_df.shape)
    
    train_dataset = CDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=config.num_workers)

    val_dataset = CDataset(val_df, eval_mode=True)
    val_loader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle=False, num_workers=config.num_workers)     
    
    test_dataset = CDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=False, num_workers=config.num_workers)         
    
    return train_loader, val_loader, test_loader
    
def build_model(config) :
    if config.arch == 'ae_basic' :
        return AutoEncoder()
    elif config.arch == 'ae_denoising' :
        return DenoisingAE()
    elif config.arch == 'ae_tied' :
        return TiedAE()
    elif config.arch == 'ae_stacked' :
        return StackedAE()
    else :
        print('please select the model architecture')
        return AutoEncoder()
    
def build_optimizer (model, config) :
    if config.optimizer == 'sgd' :
        optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    else : # adam
        optimizer = torch.optim.Adam(params=model.parameters(), lr = config.lr)
        
    return optimizer
        
        
class Trainer() :
    def __init__(self, model, optimizer, train_loader, val_loader, scheduler, config) :
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = config.device
        self.epochs = config.epochs
        self.lr = config.lr
        
        self.criterion = nn.L1Loss().to(self.device)
        
    def fit(self,) :
        self.model.to(self.device)
        best_score = 0
        for epoch in range(self.epochs) :
            self.model.train()
            train_loss = []
            
            for x in iter(self.train_loader) :
                x = x.to(self.device)
                _x = self.model(x)
                
                loss = self.criterion(x, _x)
                loss = self.criterion(x, _x)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss.append(loss.item())
                wandb.log({'train loss' : loss.item()})
                
            score = self.validation(self.model, 0.95)
            # score = round(score, 3)
            
            if self.scheduler is not None :
                self.scheduler.step(score)
            
            print(f'epoch :[{epoch}] train loss [{np.mean(train_loss):3f}] val score [{score:.3f}]')
            
            # for param_group in self.optimizer.param_groups:
            #     print(param_group['lr'])      
            
            # print(f'epoch :[{epoch}] train loss [{np.mean(train_loss)}] val score [{score}] lr [{self.scheduler.get_lr()}]')

            if best_score < score :
                best_score = score
                # torch.save(self.model.state_dict(), '../saved1/model_epoch-{}-f1-{:.3f}.pt'.format(epoch, best_score))
                torch.save(self.model.state_dict(), './saved1/best_model.pth')
            
    def validation(self, eval_model, threshold) :
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        eval_model.eval()
        pred_y = []
        true_y = []
        
        with torch.no_grad():
            for x, y in iter(self.val_loader) :
                x = x.to(self.device)
                y = y.to(self.device)
                
                _x = self.model(x)
                diff = cos(x, _x).cpu().tolist()
                batch_pred = np.where(np.array(diff) < threshold, 1, 0).tolist()
                pred_y += batch_pred
                true_y += y.tolist()
                
        f1 = f1_score(true_y, pred_y, average='macro')
        wandb.log({'f1_score' : f1})
                
        return f1        
    

def main(config) :
    
    seed_everything(config.seed)    
    
    train_loader, val_loader, test_loader = build_dataset(config)
    model = build_model(config)
    model.eval()
    
    optimizer = build_optimizer(model, config)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, threshold_mode='abs', min_lr=1e-8, verbose=True)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.2)
    
    wandb.watch(model, log='all')

    trainer = Trainer(model, optimizer, train_loader, val_loader, scheduler, config)
    
    # wandb.save('model.h5')

    trainer.fit()  
    
if __name__ == "__main__":
    args = parser.parse_args()
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    wandb.init(project='Credit Card Fraud Detection Landscape',  name=now, mode='online')
    wandb.watch_called = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config =wandb.config
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.lr = args.lr
    config.momentum = args.momentum
    config.weight_decay = args.weight_decay
    config.device = device
    config.seed = 42
    config.log_interval = 10
    config.num_workers = args.workers
    config.adam = True   
    config.data_path = args.data_path
    config.arch = args.arch
    config.optimizer = args.optimizer
    
    main(config)