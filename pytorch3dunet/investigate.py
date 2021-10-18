import importlib
import os

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.investigator import get_model

logger = utils.get_logger('UNet3DInvestigate')


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


def _get_investigator(model, loader, config):
    investigator_config = config.get('investigator', {})
    class_name = investigator_config.get('name', 'StandardInvestigator')

    m = importlib.import_module('pytorch3dunet.unet3d.investigator')
    investigator_class = getattr(m, class_name)
    
    return investigator_class(model, loader, config, **investigator_config)


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config)
    model.testing = True

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model, strict=False)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')

    for train_loader in get_train_loaders(config):
        logger.info(f"Processing '{train_loader.dataset.file_path}'...")

        output_file = config['investigator']['output_path'] # 결과 png 저장할 장소

        investigator = _get_investigator(model, train_loader, output_file, config)
        investigator.investigate()


if __name__ == '__main__':
    main()
