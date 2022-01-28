#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:43:48 2020

@author: shkim
"""
import importlib
import os
import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger('UNet3DPredict')

def _get_output_file(dataset, suffix='_predictions', output_dir=None):
    input_dir, file_name = os.path.split(dataset.file_path)
    if output_dir is None:
        output_dir = input_dir
    output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + suffix + '.h5')
    return output_file


def _get_dataset_names(config, number_of_datasets, prefix='predictions'):
    dataset_names = config.get('dest_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]


def _get_predictor(model, loader, output_file, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, loader, output_file, config, **predictor_config)

def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    # args = parser.parse_args()
    config = _load_config_yaml('/home/shkim/Libraries/pytorch-3dunet/resources/test_config_ac.yaml')
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config

def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))

def displayImage(image, name):
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(name)
    ax = fig.add_subplot(1,1,1)
    ax.imshow(image, cmap="gray")
    fig.tight_layout()
    # plt.show()
    plt.savefig('/home/shkim/Libraries/pytorch-3dunet/fig_filter/' + name)
    plt.close(fig)
    
def displayImage3(image1, image2, image3, name):
    fig = plt.figure(figsize=(4,12))
    fig.suptitle(name)
    ax = fig.add_subplot(3,1,1)
    ax.imshow(image1, cmap="gray")
    ax = fig.add_subplot(3,1,2)
    ax.imshow(image2, cmap="gray")
    ax = fig.add_subplot(3,1,3)
    ax.imshow(image3, cmap="gray")
    fig.tight_layout()
    # plt.show()
    plt.savefig('/home/shkim/Libraries/pytorch-3dunet/fig_filter/' + name)
    plt.close(fig)

class IntermediateOutput(nn.Module):
    def __init__(self, original_model, i):
        super(IntermediateOutput, self).__init__()
        self.features = nn.Sequential(*list(original_model.encoders[:-i]))
        
    def forward(self, x):
        x = self.features(x)
        return x
    
def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config)
    model.testing = True
    
    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # # use DataParallel if more than 1 GPU available
    # device = config['device']
    # if torch.cuda.device_count() > 1 and not device.type == 'cpu':
    #     model = nn.DataParallel(model)
    #     logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    # logger.info(f"Sending the model to '{device}'")
    # model = model.to(device)
    
    loaders = get_test_loaders(config)
    # imd = IntermediateOutput(model, 1)
    
    # print(model)
    with torch.no_grad():
        for c, t in enumerate(loaders):
            print('Data loaded: ', c)
            for count, x in enumerate(t.dataset):
                x = x[0]
                for lv in range(4,0,-1):
                    imd = IntermediateOutput(model, lv)
                    imd.eval()
                    res = imd.forward(x.unsqueeze(0))
                    res.detach().numpy()
                    print('Level: ', 4-lv , ' - ', res.shape)
                    for i in range(res.shape[0]):
                        for j in range(res.shape[1]):
                            print(4-lv, i, j)
                            np.save('/home/shkim/Libraries/pytorch-3dunet/fig_filter/' + str(count) + '_' + str(4-lv) + '_' + str(i) + '_' + str(j), res[i,j,:,:,:])
                            displayImage(res[i,j,:,:,int(res.shape[4]/2)], 'img' + str(count) + '_' + str(4-lv) + '_' + str(i) + '_' + str(j) + '.png')
    
    # for name, param in model.named_parameters():
    #     print(name, '\t\t\t', param.shape)
    #     if len(param.shape) == 5 and param.shape[2] == 3:
    #         for i in range(param.shape[0]):
    #             for j in range(param.shape[1]):
    #                 displayImage3(param[i,j,0,:,:].detach().numpy(),
    #                               param[i,j,1,:,:].detach().numpy(),
    #                               param[i,j,2,:,:].detach().numpy(), name + str(i) + '_' + str(j) + 'dim0.png')
    #                 displayImage3(param[i,j,:,0,:].detach().numpy(),
    #                               param[i,j,:,1,:].detach().numpy(),
    #                               param[i,j,:,2,:].detach().numpy(), name + str(i) + '_' + str(j) + 'dim1.png')
    #                 displayImage3(param[i,j,:,:,0].detach().numpy(),
    #                               param[i,j,:,:,1].detach().numpy(),
    #                               param[i,j,:,:,2].detach().numpy(), name + str(i) + '_' + str(j) + 'dim2.png')

    # output_dir = config['loaders'].get('output_dir', None)
    # if output_dir is not None:
        # os.makedirs(output_dir, exist_ok=True)
    #     logger.info(f'Saving predictions to: {output_dir}')

    # for test_loader in get_test_loaders(config):
    #     logger.info(f"Processing '{test_loader.dataset.file_path}'...")

    #     output_file = _get_output_file(dataset=test_loader.dataset, output_dir=output_dir)

    #     predictor = _get_predictor(model, test_loader, output_file, config)
    #     # run the model prediction on the entire dataset and save to the 'output_file' H5
    #     predictor.predict()


if __name__ == '__main__':
    main()
