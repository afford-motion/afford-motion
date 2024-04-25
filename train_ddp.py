import os
import hydra
import torch
import random
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.distributed import DistributedSampler

from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, Board
from utils.training import TrainLoop
from utils.misc import compute_repr_dimesion

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ Main function

    Args:
        cfg: configuration dict
    """
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    
    ## set rank and device
    cfg.gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(cfg.gpu)
    device = torch.device('cuda', cfg.gpu)
    torch.distributed.init_process_group(backend='nccl')

    ## set output logger and plot board
    if cfg.gpu == 0:
        logger.remove(handler_id=0) # remove default handler
        mkdir_if_not_exists(cfg.log_dir)
        mkdir_if_not_exists(cfg.ckpt_dir)
        mkdir_if_not_exists(cfg.eval_dir)

        logger.add(cfg.log_dir + '/runtime.log')
        Board().create_board(cfg.platform, project=cfg.project, log_dir=cfg.log_dir)

        ## Begin training progress
        logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
        logger.info('[Train] ==> Beign training..')
    
    # prepare training dataset
    train_dataset = create_dataset(cfg.task.dataset, cfg.task.train.phase, gpu=cfg.gpu)
    if cfg.gpu == 0:
        logger.info(f'Load train dataset size: {len(train_dataset)}')
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    train_dataloader = train_dataset.get_dataloader(
        sampler=train_sampler,
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        pin_memory=True,
    )

    ## create model and optimizer
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cfg.gpu], output_device=cfg.gpu, find_unused_parameters=True, broadcast_buffers=False)

    ## start training
    TrainLoop(
        cfg=cfg.task.train,
        model=model,
        diffusion=diffusion,
        dataloader=train_dataloader,
        device=device,
        save_dir=cfg.ckpt_dir,
        gpu=cfg.gpu,
        is_distributed=True,
    ).run_loop()

    ## Training is over!
    if cfg.gpu == 0:
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