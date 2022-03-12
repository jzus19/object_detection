from omegaconf import DictConfig

import pydoc
import logging


__all__ = [
    'cls_init',
]

logger = logging.getLogger(__name__)

def _locate(path: str):
    result = pydoc.locate(path)
    if result is None:
        raise ValueError(f"failed to find class: {path}")
    return result

def cls_init(class_init, *args, **extra_kwargs):
    try:
        if class_init is None:
            return None

        elif type(class_init) is str:
            cls = _locate(class_init)
            return cls()

        else:
            cls = _locate(class_init['cls'])
            kwargs = class_init.get('args', dict())

            return cls(
                *args,
                **extra_kwargs,
                **kwargs
            )
    except:
        logger.exception(f"Could not instantiate from config {class_init}")
        raise    