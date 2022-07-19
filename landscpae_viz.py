# base code is from below link
# https://gitlab.com/qbeer/loss-landscape/-/blob/main/run_imagenet.py

import torch
import numpy as np
import os
from loss_landscape.models import load_model, model_ids
from loss_landscape.landscape_utils import init_directions, init_network
import argparse

import matplotlib.pyplot as plt

def run_landscape_gen(args):
    BATCH_SIZE = args.batch_size
    RESOLUTION = args.resolution

    dataset = torchvision.datasets.ImageFolder(root=args.path_to_imagenetv2,
                                            transform=T.Compose([T.Resize(256),
                                                                 T.CenterCrop(224),
                                                                 T.ToTensor(),
                                                                 T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    for model_id in model_ids:
        print(f'Testing {model_id}')

        if os.path.exists(f'results/{model_id}_contour_bs_{BATCH_SIZE}_res_{RESOLUTION}_imagenetv2.png'):
            continue

        noises = init_directions(load_model(model_id))

        crit = torch.nn.CrossEntropyLoss()

        A, B = np.meshgrid(np.linspace(-1, 1, RESOLUTION),
                        np.linspace(-1, 1, RESOLUTION), indexing='ij')

        loss_surface = np.empty_like(A)

        for i in range(RESOLUTION):
            for j in range(RESOLUTION):
                total_loss = 0.
                n_batch = 0
                alpha = A[i, j]
                beta = B[i, j]
                net = init_network(load_model(model_id), noises, alpha, beta).to('cuda')
                for batch, labels in dataloader:
                    batch = batch.to('cuda')
                    labels = labels.to('cuda')
                    with torch.no_grad():
                        preds = net(batch)
                        loss = crit(preds, labels)
                        total_loss += loss.item()
                        n_batch += 1
                loss_surface[i, j] = total_loss / (n_batch * BATCH_SIZE)
                del net, batch, labels
                print(f'alpha : {alpha:.2f}, beta : {beta:.2f}, loss : {loss_surface[i, j]:.2f}')
                torch.cuda.empty_cache()

        plt.figure(figsize=(10, 10))
        plt.contour(A, B, loss_surface)
        plt.savefig(f'results/{model_id}_contour_bs_{BATCH_SIZE}_res_{RESOLUTION}_imagenetv2.png', dpi=100)
        plt.close()

        np.save(f'{model_id}_xx_imagenetv2.npy', A)
        np.save(f'{model_id}_yy_imagenetv2.npy', B)
        np.save(f'{model_id}_zz_imagenetv2.npy', loss_surface)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to generate loss landscape.')
    parser.add_argument('--path_to_imagenetv2', default='/mnt/storage/datasets/imagenetv2-top-images-format-val/', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--resolution', default=25, type=int)

    args = parser.parse_args()

    run_landscape_gen(args)
