import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3dunet.unet3d.scheduler import CosineAnnealingWarmUpRestarts

from pytorch3dunet.datasets.utils import get_train_loaders, get_dejavu_loader
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.trainer import UNet3DTrainer
from pytorch3dunet.unet3d.utils import get_logger, get_tensorboard_formatter
from pytorch3dunet.unet3d.utils import get_number_of_learnable_parameters

logger = get_logger('UNet3DTrain')


def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)
    skip_train_validation = trainer_config.get('skip_train_validation', False)
    accumulation_iters = trainer_config.get('accumulation_iters', 1)

    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.get('tensorboard_formatter', None))

    if resume is not None:
        # continue training from a given checkpoint
        return UNet3DTrainer.from_checkpoint(resume, model,
                                             optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, loaders, tensorboard_formatter=tensorboard_formatter,
                                             accumulation_iters=accumulation_iters)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return UNet3DTrainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, device=config['device'], loaders=loaders,
                                             max_num_epochs=trainer_config['epochs'],
                                             max_num_iterations=trainer_config['iters'],
                                             validate_after_iters=trainer_config['validate_after_iters'],
                                             log_after_iters=trainer_config['log_after_iters'],
                                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                                             tensorboard_formatter=tensorboard_formatter,
                                             skip_train_validation=skip_train_validation,
                                             accumulation_iters=accumulation_iters)
    else:
        # start training from scratch
        return UNet3DTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                             config['device'], loaders, trainer_config['checkpoint_dir'],
                             max_num_epochs=trainer_config['epochs'],
                             max_num_iterations=trainer_config['iters'],
                             validate_after_iters=trainer_config['validate_after_iters'],
                             log_after_iters=trainer_config['log_after_iters'],
                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                             tensorboard_formatter=tensorboard_formatter,
                             skip_train_validation=skip_train_validation,
                             accumulation_iters=accumulation_iters)


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        lr_config['optimizer'] = optimizer
        modules = ['torch.optim.lr_scheduler', 'pytorch3dunet.unet3d.scheduler']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz(**lr_config)
        raise RuntimeError(f'Unsupported lr_scheduler class: {class_name}')


def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)

    # torch.cuda.set_enabled_lms(True)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    model = get_model(config)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    
    # Create data loaders
    loaders = get_train_loaders(config)
    logger.info(f'The size of train/val dataset is {len(loaders["train"])}/{len(loaders["val"])}')
    
    # Create the optimizer
    optimizer = _create_optimizer(config, model)
    
    # Create loss criterion
    loss_criterion = get_loss_criterion(config)

    if config['loss']['name'] == 'EWCLoss':
        # loss_criterion += model.ewc_loss
        loader = get_dejavu_loader(config)
        if torch.cuda.device_count() > 1 and not device.type == 'cpu':
            model.module.estimate_fisher(loader, device, 1)
        else:
            model.estimate_fisher(loader, device, 1)
    # if torch.cuda.device_count() > 1 and not device.type == 'cpu':
    #     model = nn.DataParallel(model)
        
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
