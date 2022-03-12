from omegaconf import DictConfig
import hydra
import logging 
from pathlib import Path
import os

import pytorch_lightning as pl
import sys
#custom libraries
from src.utils import cls_init
from src.system import Model
from src.data import get_loaders

logger = logging.getLogger(__name__)

def train(cfg: DictConfig) -> None:
    ckpt_path = cfg.train.ckpt_path
    #if checkpoint doesn't exist --> error
    if (ckpt_path is not None) and (not Path(ckpt_path).exists()):
        logger.warning(f"Not using missing checkpoint {ckpt_path}, starting from scratch...")
        ckpt_path = None

    callbacks = [cls_init(callback) for callback in cfg.train.callbacks.values()]
    logger_pl = cls_init(cfg.train.logger)

    trainer = pl.Trainer(
        **cfg.train.train_params,
        callbacks=callbacks,
        logger=logger_pl,
        profiler="simple",
        resume_from_checkpoint=ckpt_path
    )

    model = Model(config=cfg)
    
    if cfg.train.train_params.auto_lr_find:
        trainer.tune(model)

        if resume_from_checkpoint is not None:
            logger.info(f"Resuming training from checkpoint {resume_from_checkpoint}")
    else:
        model.init_pretrain_modules()

    train_dataloader, val_dataloader = get_loaders(
        train_cls=cfg.data.train,
        val_cls=cfg.data.val,
        batch_size=cfg.train.batch_size,
    )

    trainer.fit(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, model=model)

    
