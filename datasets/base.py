from typing import Dict
from omegaconf import DictConfig
from torch.utils.data import Dataset
from utils.registry import Registry
DATASET = Registry('Dataset')

def create_dataset(cfg: DictConfig, phase: str, **kwargs: Dict) -> Dataset:
    """ Create a `torch.utils.data.Dataset` object from configuration.

    Args:
        cfg: configuration object, dataset configuration
        phase: phase string, can be 'train' and 'test'
    
    Return:
        A Dataset object that has loaded the designated dataset.
    """
    return DATASET.get(cfg.name)(cfg, phase, **kwargs)