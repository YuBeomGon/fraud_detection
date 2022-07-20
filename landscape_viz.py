# base code is from below link
# https://gitlab.com/qbeer/loss-landscape/-/blob/main/run_imagenet.py

import torch
import torch.nn as nn
import numpy as np
import os
from loss_landscape.landscape_utils import init_directions, init_network
import argparse
from datetime import datetime
import wandb

parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Landscape')
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


import matplotlib.pyplot as plt
from train import build_dataset, build_model, seed_everything

# model_ids = ['ae_basic']
# model_ids = ['ae_basic', 'ae_stacked']

def load_model(model, config):
    path = 'saved1/' + config.arch + '/best_model.pth'
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def run_landscape_gen(config):
    
    seed_everything(config.seed) 
    
    _, val_loader, _ = build_dataset(config)
    
    for model_id in config.model_ids:
        print(f'Testing {model_id}')
        config.arch = model_id
        model = build_model(config.arch)
        model.eval() 

        if os.path.exists(f'results/{model_id}_contour_bs_{config.batch_size}_res_{config.resolution}_.png'):
            continue

        noises = init_directions(load_model(model, config))

        criterion = nn.L1Loss()

        A, B = np.meshgrid(np.linspace(-1, 1, config.resolution),
                        np.linspace(-1, 1, config.resolution), indexing='ij')

        loss_surface = np.empty_like(A)

        for i in range(config.resolution):
            for j in range(config.resolution):
                total_loss = 0.
                n_batch = 0
                alpha = A[i, j]
                beta = B[i, j]
                net = init_network(load_model(model, config), noises, alpha, beta).to('cuda')
                for batch, label in iter(val_loader):
                    batch = batch.to('cuda')
                    label = label.to('cuda')
                    with torch.no_grad():
                        preds = net(batch)
                        loss = criterion(batch, preds)
                        # loss = criterion(label, preds)
                        total_loss += loss.item()
                        n_batch += 1
                loss_surface[i, j] = total_loss / (n_batch * config.batch_size)
                del net, batch, preds
                print(f'alpha : {alpha:.2f}, beta : {beta:.2f}, loss : {loss_surface[i, j]:.2f}')
                torch.cuda.empty_cache()

        plt.figure(figsize=(10, 10))
        plt.contour(A, B, loss_surface)
        plt.savefig(f'results/{model_id}_contour_bs_{config.batch_size}_res_{config.resolution}.png', dpi=100)
        plt.close()

        np.save(f'{model_id}_xx_card.npy', A)
        np.save(f'{model_id}_yy_card.npy', B)
        np.save(f'{model_id}_zz_card.npy', loss_surface)


if __name__ == '__main__':
    args = parser.parse_args()
    now = datetime.now().strftime('%Y%m%d_%H%M%S')    
    
    wandb.init(project='Credit Card Fraud Detection Landscape',  name=now, mode='online')
    wandb.watch_called = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config =wandb.config
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.device = device
    config.num_workers = args.workers
    config.data_path = args.data_path
    config.arch = args.arch
    config.seed = 42
    config.resolution = 100 # sweep 100*100 for visualizing
    config.model_ids = ['ae_basic']

    run_landscape_gen(config)
