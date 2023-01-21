import os
from comet_ml import Artifact, Experiment
import torch
from ml_tools.data.data_loader import ImageFolderDataModule
from ml_tools.utils.utils import generate_model_config, read_cfg, get_optimizer, get_device, generate_hyperparameters
from ml_tools.models import timm_base_model
from ml_tools.trainer import Trainer
from ml_tools.utils.schedulers import CosineAnealingWithWarmUp
from ml_tools.utils.callbacks import CustomCallback
from torchvision import transforms
import torch.nn as nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Argument for train the model")
    parser.add_argument('-cfg', '--config', type=str, help="Path to config yaml file")
    return parser

if __name__ == "__main__":

    args = parse_args()
    cfg = read_cfg(cfg_file=args.config)
    hyperparameters = generate_hyperparameters(cfg)

    print("Training Process Start")
    logger = Experiment(api_key=cfg['logger']['api_key'], 
            project_name=cfg['logger']['project_name'],
            workspace=cfg['logger']['workspace']) # logger for track model in Comet ML
    artifact = Artifact("", "Model")
    print("Comet Logger has successfully loaded.")

    device = get_device(cfg)
    print(f"{str(device)} has choosen.")

    kwargs = dict(pretrained=cfg['model']['pretrained'], output_class=cfg['model']['num_classes'], multilabel=cfg['model']['multilabel'])
    network = timm_base_model.TimmBaseClassificationModel(cfg['model']['base'], **kwargs)
    print(network)
    print(f"Network {cfg['model']['base']} succesfully loaded.")

    optimizer = get_optimizer(cfg, network)
    print(f"Optimizer has been defined.")

    lr_scheduler = CosineAnealingWithWarmUp(optimizer, 
        first_cycle_steps=250, 
        cycle_mult=0.5,
        max_lr=1e-2, 
        min_lr=cfg['train']['lr'], 
        warmup_steps=100, 
        gamma=0.5)

    print(f"Scheduler has been defined.")

    criterion = nn.CrossEntropyLoss()
    print(f"Criterion has been defined")

    train_val_transforms = transforms.Compose([
        transforms.RandomRotation(cfg['dataset']['augmentation']['rotation_range']),
        transforms.Resize(cfg['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['std'])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(cfg['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['std'])
    ])

    dataset = ImageFolderDataModule(cfg, train_val_transforms=train_val_transforms, test_transforms=test_transforms)
    print(f"Dataset successfully loaded.")
    
    cb_config = dict(
        checkpoint_path=cfg['output_dir'],
        patience=cfg['custom_cb']['patience'],
        metric=cfg['custom_cb']['metric'],
        mode=cfg['custom_cb']['mode']
    )

    custom_cb = CustomCallback(**cb_config)
    print(f"Custom CB Initialized")
    logger.log_parameters(hyperparameters)
    print("Parameters has been Logged")
    generate_model_config(cfg)
    print("Model config has been generated")

    try:
        if cfg['model']['pretrained_path'] != 'None':
            net_state_dict = torch.load(cfg['model']['pretrained_path'], map_location=device)
            network = network.load_state_dict(state_dict=net_state_dict)
        
        if cfg['optimizer']['pretrained_path'] != 'None':
            opt_state_dict = torch.load(cfg['optimizer']['pretrained_path'], map_location=device)
            optimizer = optimizer.load_state_dict(opt_state_dict)
        
        print("Pretrained has been loaded...")
    except:
        print("Pretrained Failed to Load.. Continues training process using weight from imageNet")
    
    trainer = Trainer(cfg, network, optimizer, criterion, dataset, device, callbacks=custom_cb, lr_scheduler=lr_scheduler, logger=logger)

    trainer.train()
    best_model_path = os.path.join(cfg['output_dir'], 'best_model.pth')
    best_optimizer_path = os.path.join(cfg['output_dir'], 'best_optimizer.pth')
    final_model_path = os.path.join(cfg['output_dir'], 'final_model.pth')
    final_optimizer_path = os.path.join(cfg['output_dir'], 'final_optimizer.pth')
    model_cfg_path = os.path.join(cfg['output_dir'], 'model-config.yaml')
    artifact.add(best_model_path)
    artifact.add(best_optimizer_path)
    artifact.add(final_model_path)
    artifact.add(final_optimizer_path)
    artifact.add(model_cfg_path)
    logger.log_artifact(artifact=artifact)