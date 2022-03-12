# typing imports
from typing import List, Dict, Tuple
from torch._C import Value
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim import Optimizer
from omegaconf import DictConfig

# generic imports
import logging
from omegaconf import OmegaConf
from operator import attrgetter
import sys
# torch imports
import pytorch_lightning as pl
import torch
from torch import nn
import torch.distributed as dist
import torchvision.models as models
# custom imports
from src.utils import cls_init

__all__ = ['Model']

logger = logging.getLogger(__name__)

import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


def get_train_efficientdet(cfg):
    config = get_efficientdet_config('tf_efficientdet_d5')
    config.image_size = [264,264]
    config.norm_kwargs=dict(eps=.001, momentum=.01)

    net = EfficientDet(config, pretrained_backbone=cfg.model.model_params.pretrained)
    # checkpoint = torch.load('../input/efficientdet/efficientdet_d5-ef44aea8.pth')

    # net.load_state_dict(checkpoint)
    net.reset_head(num_classes=cfg.model.model_params.num_classes)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    return DetBenchTrain(net, config)

class Model(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = get_train_efficientdet(config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        # return self.step_model(self.model(batch), mode='train')
        loss = self(batch).sum()
        return {"loss": loss}

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        # return self.step_model(self.model(batch), mode='val')
        loss = self(batch).sum()
        return {"loss": loss}
        

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        config = self.hparams.optimization
        param_groups = list()
        used_params = set()
        if 'param_groups' in config:
            for encoder, param_group in config.param_groups.items():
                group_parameters = list()
                for path in param_group.modules:
                    module = attrgetter(path)(self.encoders[encoder])
                    parameters = [x for x in module.parameters() if x.requires_grad]
                    if len(parameters) == 0:
                        raise ValueError("No params covered")
                    conflicting_parameters = set(parameters) & set(used_params)
                    if len(conflicting_parameters) > 0:
                        raise ValueError(f"Some parametrs are used in multiple config groups: {conflicting_parameters}")
                    used_params.update(parameters)
                    group_parameters.extend(parameters)
                param_groups.append({
                    'params': group_parameters,
                    **param_group.params
                })

        param_groups.append({
            'params': list(set(self.parameters()) - used_params)
        })

        optimizer = cls_init(config.optimizer, params=param_groups)
        lr_scheduler = OmegaConf.to_container(config.lr_scheduler, resolve=True)
        lr_scheduler['scheduler'] = cls_init(lr_scheduler['scheduler'], optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def init_pretrain_modules(self):
        if 'pretrain' not in self.hparams:
            return
        ckpt_path = self.hparams.pretrain.checkpoint_path
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        self.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded pretrained state from checkpoint file {ckpt_path}")