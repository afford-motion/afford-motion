import hydra
import torch
import random
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, Board
from utils.training import TrainLoop
from utils.misc import compute_repr_dimesion

def train(cfg: DictConfig) -> None:
    """ Begin training with this function

    Args:
        cfg: configuration dict
    """
    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    # prepare training dataset
    train_dataset = create_dataset(cfg.task.dataset, cfg.task.train.phase, gpu=cfg.gpu)
    logger.info(f'Load train dataset size: {len(train_dataset)}')
    
    train_dataloader = train_dataset.get_dataloader(
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    ## create model and optimizer
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    ## start training
    TrainLoop(
        cfg=cfg.task.train,
        model=model,
        diffusion=diffusion,
        dataloader=train_dataloader,
        device=device,
        save_dir=cfg.ckpt_dir,
        gpu=cfg.gpu,
    ).run_loop()

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ Main function

    Args:
        cfg: configuration dict
    """
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)

    ## set output logger and plot board
    mkdir_if_not_exists(cfg.log_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    mkdir_if_not_exists(cfg.eval_dir)

    logger.add(cfg.log_dir + '/runtime.log')
    Board().create_board(cfg.platform, project=cfg.project, log_dir=cfg.log_dir) # call one time

    ## Begin training progress
    logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
    logger.info('[Train] ==> Beign training..')

    train(cfg) # training portal

    ## Training is over!
    Board().close() # close board
    logger.info('[Train] ==> End training..')

if __name__ == '__main__':
    SEED = 2023
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    
    main()