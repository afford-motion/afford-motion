import os, glob, hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted

from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, time_str
from utils.evaluate import create_evaluator
from utils.training import load_ckpt
from utils.misc import compute_repr_dimesion

def test(cfg: DictConfig) -> None:
    """ Begin testing with this function

    Args:
        cfg: configuration dict
    """
    test_dir = os.path.join(cfg.eval_dir, 'test-' + time_str(Y=False))
    mkdir_if_not_exists(test_dir)
    logger.add(os.path.join(test_dir, 'test.log'))
    logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
    logger.info('[Test] ==> Beign testing..')

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    # prepare testing dataset
    test_dataset = create_dataset(cfg.task.dataset, 'test', gpu=cfg.gpu, **cfg.task.test)
    logger.info(f'Load test dataset size: {len(test_dataset)}')
    
    test_dataloader = test_dataset.get_dataloader(
        batch_size=cfg.task.test.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.test.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    ## create model and load checkpoint
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    ckpts = natsorted(glob.glob(os.path.join(cfg.exp_dir, 'ckpt', 'model*.pt')))
    assert len(ckpts) > 0, 'No checkpoint found.'
    load_ckpt(model, ckpts[-1])
    logger.info(f'Load checkpoint from {ckpts[-1]}')

    ## create evaluator
    evaluator = create_evaluator(cfg.task, device=device)
    
    ## sample
    model.eval()
    sample_fn = diffusion.p_sample_loop

    B = test_dataloader.batch_size
    sample_list = []
    k_samples_list = []
    if evaluator.k_samples > 0:
        k_samples_idxs = list(range(evaluator.num_k_samples // B)) # first len(k_samples_idxs) batches will be used for k samples (repeat_times = k_samples)
    else:
        k_samples_idxs = []
    logger.info(f'k_samples_idxs: {k_samples_idxs}')

    for i, data in enumerate(test_dataloader):
        logger.info(f"batch index: {i}, is k_sample_batch: {i in k_samples_idxs}, case index: {data['info_index']}")
        x = data['x']

        x_kwargs = {}
        if 'x_mask' in data:
            x_kwargs['x_mask'] = data['x_mask'].to(device)

        for key in data:
            if key.startswith('c_') or key.startswith('info_'):
                if torch.is_tensor(data[key]):
                    x_kwargs[key] = data[key].to(device)
                else:
                    x_kwargs[key] = data[key]

        use_k_sample = i in k_samples_idxs
        repeat_times = evaluator.k_samples if use_k_sample else 1
        
        sample_list_np = []
        k_samples_list_np = []
        for k in range(repeat_times):
            if cfg.model.name.startswith('CMDM'):
                ## if test with CMDM, the input c_pc_contact contains k samples, 
                ## so we need remove this item in x_kwargs, and use the k-th contact map
                x_kwargs['c_pc_contact'] = data['c_pc_contact'][:, k, :, :].to(device)

            sample = sample_fn(
                model,
                x.shape,
                clip_denoised=False,
                noise=None,
                model_kwargs=x_kwargs,
                progress=True,
            )

            if k == 0:
                for bsi in range(B):
                    sample_list_np.append(sample[bsi].cpu().numpy())
            
            if use_k_sample:
                for bsi in range(B):
                    k_samples_list_np.append(sample[bsi].cpu().numpy())
        
        ## 1 sample
        for bsi in range(B):
            res_dict = {'sample': sample_list_np[bsi]}
            for key in data:
                if torch.is_tensor(data[key]):
                    res_dict[key] = data[key][bsi].cpu().numpy()
                else:
                    res_dict[key] = data[key][bsi]
            sample_list.append(res_dict)
        
        ## k samples
        if use_k_sample:
            for bsi in range(B):
                res_dict = {'k_samples': np.stack(k_samples_list_np[bsi::B])}
                for key in data:
                    if torch.is_tensor(data[key]):
                        res_dict[key] = data[key][bsi].cpu().numpy()
                    else:
                        res_dict[key] = data[key][bsi]
                k_samples_list.append(res_dict)
        
        ## stop evaluation if reach the max number of samples
        if i + 1 >= evaluator.eval_nbatch:
            break
    
    ## compute metrics
    evaluator.evaluate(sample_list, k_samples_list, test_dir, test_dataloader, device=device)
    evaluator.report(test_dir)

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ Main function

    Args:
        cfg: configuration dict
    """
    ## setup random seed
    SEED = cfg.seed
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    ## compute modeling dimension
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)

    ## set output logger
    mkdir_if_not_exists(cfg.log_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    mkdir_if_not_exists(cfg.eval_dir)

    test(cfg) # testing portal

if __name__ == '__main__':
    import torch
    import random
    import numpy as np
    
    main()