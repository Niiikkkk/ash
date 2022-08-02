#!/usr/bin/env python

import argparse
import os

import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_factory import build_dataset, get_num_classes
from models.model_factory import build_model
from utils.utils import is_debug_session

os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'


def wandb_plot(x_values, y_values, x_axis="x", y_axis="y", title="Custom plot", type="line"):
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=[x_axis, y_axis])
    if type == 'line':
        plot = wandb.plot.line(table, x_axis, y_axis, title=title)
    elif type == 'scatter':
        plot = wandb.plot.scatter(table, x_axis, y_axis, title=title)
    else:
        plot = None
    return plot


def pruning(use_gpu, use_tqdm):
    """
    Recording safe regions on in-distribution datasets (cifar10 or cifar100 or svhn)

    @param args:
    @return:
    """
    config = {
        'dataset_name': 'imagenet',
        'model_name': 'resnet50',
        'train_restore_file': 'resnet50-19c8e357.pth',
        'batch_size': 200,
    }

    # init wandb
    wandb.init(project="ash",
               dir=os.getenv("LOG"),
               config=config,
               name='LAYER1',
               tags=['PRUNING'])

    # construct the model
    num_classes = get_num_classes(config['dataset_name'])
    model, transform = build_model(config['model_name'], num_classes=num_classes)
    checkpoint = os.path.join(os.getenv('MODELS'), config['train_restore_file'])
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # setup dataset
    dataset = build_dataset(config['dataset_name'], transform, train=False)

    kwargs = {}
    if torch.cuda.is_available() and not is_debug_session():
        kwargs = {'num_workers': 4, 'pin_memory': True}

    if use_gpu:
        model = model.cuda()

    thresholds = np.array([70, 80, 85, 90, 92, 94, 96, 98, 99, 100])
    scores = []
    for t in range(thresholds.shape[0]):
        if use_tqdm:
            progress_bar = tqdm(total=len(dataset), leave=False)

        # apply ash
        os.environ['ash_method'] = f'ash_p@{thresholds[t]}'

        with torch.no_grad():
            dataloader = DataLoader(dataset,
                                    batch_size=config.get("batch_size"),
                                    **kwargs)
            gt = list()
            p = list()
            for i, samples in enumerate(dataloader):
                # if i >= 2:
                #     break
                images = samples[0]
                labels = samples[1]

                # Create non_blocking tensors for distributed training
                if use_gpu:
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)

                gt.append(labels.detach().cpu().numpy())
                p.append(predicted.detach().cpu().numpy())

                if use_tqdm:
                    progress_bar.update(images.size(0))

            if use_tqdm:
                progress_bar.close()

            gt = np.concatenate(gt)
            p = np.concatenate(p)
            acc1 = accuracy_score(gt, p)
            scores.append(acc1)
            wandb.log({
                'acc': acc1,
                'p': thresholds[t]
            })

        plot = wandb_plot(thresholds, scores, x_axis="PRUNING_PERCENTAGE", y_axis="ACCURACY", title="ACCURACY DEGRADATION")
        wandb.log({
            f'plt/accuracy_degradation': plot,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU")
    parser.add_argument("--use-tqdm", action="store_true", default=False, help="Enables progress bar")
    args = parser.parse_args()
    pruning(args.use_gpu, args.use_tqdm)
