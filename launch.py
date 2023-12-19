import sys
import argparse
import os
import time
import logging
from datetime import datetime

import datasets
import systems
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from utils.callbacks import (
    CodeSnapshotCallback,
    ConfigSnapshotCallback,
    CustomProgressBar,
)
from utils.misc import load_config
from loguru import logger
import torch
import wandb

wandb.login(key="3150d2d7cfb2243369255497308e5457525529be")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")
    parser.add_argument(
        "--resume", default=None, help="path to the weights to be resumed"
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--predict", action="store_true")

    parser.add_argument("--exp_dir", default="./exp")
    parser.add_argument("--runs_dir", default="./runs")
    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    args, extras = parser.parse_known_args()
    return args, extras

def set_environtment(gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def get_environtment(gpu):
    n_gpus = len(gpu.split(","))
    return n_gpus

def load_info_config(args, extras):
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config') 
    
    if "seed" not in config:
        config.seed = int(time.time() * 1000) % 1000 
    pl.seed_everything(config.seed)

    return config

def set_logger(args):
    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

def get_info(is_train, runs_dir, config):
    callbacks = []
    if is_train:
        callbacks += [
            ModelCheckpoint(dirpath=config.ckpt_dir, **config.checkpoint),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(config.code_dir, use_version=False),
            ConfigSnapshotCallback(config, config.config_dir, use_version=False),
            CustomProgressBar(refresh_rate=1),
        ]

    loggers = []
    if is_train:
        loggers += [
            TensorBoardLogger(
                runs_dir, name=config.name, version=config.trial_name
            ),
            CSVLogger(config.exp_dir, name=config.trial_name, version="csv_logs"),
        ]
    return loggers, callbacks

def init_log(configs):
    model_name = configs.model.name 
    data_name = configs.dataset.scene
    mode_run = configs.dataset.name 
    
    config = dict(configs)
    
    tags = []
    if int(configs.system.loss.lambda_distortion) > 0 
        tags.append("Lambda distortion")

    wandb.init(
        tags=tags,
        project="NeRF-MRE",
        config=config
    )


def main():
    torch.set_float32_matmul_precision('high')
    args, extras = get_args()
    set_environtment(gpu = args.gpu)
    n_gpus = get_environtment(gpu = args.gpu)
    config = load_info_config(args, extras)
    init_log(config)

    logger = set_logger(args)
    loggers, callbacks = get_info(args.train, args.runs_dir, config)
    strategy = "ddp"
    # strategy = 'ddp_find_unused_parameters_false'

    trainer = Trainer(
        devices=n_gpus,
        accelerator="gpu",
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        **config.trainer
    )
    dm = datasets.make(config.dataset.name, config.dataset)
    
    system = systems.make(
        config.system.name,
        config,
        load_from_checkpoint=None if not args.resume_weights_only else args.resume,
    )
    
    if args.train:
        if args.resume and not args.resume_weights_only:
            trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
    # validation
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume)
    # test
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=args.resume)
    # predict
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=args.resume)

if __name__ == "__main__":
    main()
