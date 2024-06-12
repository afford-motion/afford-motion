# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import os
import functools
import torch
import torch.nn as nn
from loguru import logger

from utils.io import Board
from diffusion.resample import uniform_sampling

class TrainLoop:
    def __init__(self, *, cfg, model, diffusion, dataloader, **kwargs) -> None:
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader

        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0
        self.is_distributed = kwargs['is_distributed'] if 'is_distributed' in kwargs else False

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        ## set optimizer
        params = []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for optimization.')
        if self.gpu == 0:
            logger.info(f'{len(params)} parameters for optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
        
        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()
        
    def _load_and_sync_parameters(self):
        """ Load model from checkpoint if provided for resuming. """
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
        
    def _load_optimizer_state(self):
        """ Load optimizer state from checkpoint if provided for resuming. """
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """ Save model and optimizer state. """
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            if 'scene_model' in key or 'clip_model' in key or 'text_model' in key or 'bert_model' in key:
                continue

            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"opt.pt"), "wb") as f: # only save the last optimizer state for saving space
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Model saved! [Step: {self.step:06d}]')
    
    def _freeze_scene_model_batchnorm(self):
        """ Freeze batchnorm in scene model if the model has scene model. """
        if hasattr(self.model, 'scene_model') and self.model.freeze_scene_model :
            for m in self.model.scene_model.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.eval()

    def run_loop(self):
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            self._freeze_scene_model_batchnorm() # freeze batchnorm in scene model if the model has scene model
            if self.is_distributed:
                self.dataloader.sampler.set_epoch(epoch)
            for it, data in enumerate(self.dataloader): 
                x = data['x'].to(self.device)

                x_kwargs = {}
                if 'x_mask' in data:
                    x_kwargs['x_mask'] = data['x_mask'].to(self.device)
                
                for key in data:
                    if key.startswith('c_') :
                        if torch.is_tensor(data[key]):
                            x_kwargs[key] = data[key].to(self.device)
                        else:
                            x_kwargs[key] = data[key]

                ## one step optimization
                self.optimizer.zero_grad()

                t = uniform_sampling(x.shape[0], self.device, self.diffusion.num_timesteps)
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    x,
                    t,
                    model_kwargs=x_kwargs,
                    epoch=epoch
                )
                terms = compute_losses()
                loss = terms['loss'].mean()
                loss.backward()

                self.optimizer.step()
                self._anneal_lr()
                
                ## log and save
                ## log with loguru, plot with Board
                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    ## log with loguru
                    losses = {key: terms[key].mean().item() for key in terms}

                    logger.info(
                        f"[TRAIN] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | Step: {self.step:7d} | Loss: {losses['loss']:8.5f}"
                    )

                    ## plot with Board
                    write_dict = {'step': self.step, 'train/epoch': epoch}
                    for key in losses:
                        write_dict[f'train/{key}'] = losses[key]
                    Board().write(write_dict)

                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    ## save model
                    self._save()
                
                ## update step and check max steps
                self.step += 1
                if self.step > self.max_steps:
                    return

class CVAETrainLoop:
    def __init__(self, *, cfg, model, dataloader, **kwargs) -> None:
        """ Customized training loop for HUMANISE CVAE
        """
        self.model = model
        self.dataloader = dataloader

        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        ## set optimizer
        tune_params, train_params = [], []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if 'scene_model' in n:
                    tune_params.append(p)
                else:
                    train_params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for optimization.')

        if self.gpu == 0:
            logger.info(f'{len(tune_params) + len(train_params)} parameters for optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
        
        self.optimizer = torch.optim.Adam(
            [
                {'params': tune_params, 'lr': self.lr * 0.1},
                {'params': train_params}
            ],
            lr=self.lr
        )
        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()
        
    def _load_and_sync_parameters(self):
        """ Load model from checkpoint if provided for resuming. """
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
        
    def _load_optimizer_state(self):
        """ Load optimizer state from checkpoint if provided for resuming. """
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """ Save model and optimizer state. """
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            if 'clip_model' in key or 'text_model' in key or 'bert_model' in key:
                continue

            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"opt.pt"), "wb") as f: # only save the last optimizer state for saving space
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Model saved! [Step: {self.step:06d}]')

    def run_loop(self):
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for it, data in enumerate(self.dataloader): 
                x = data['x'].to(self.device)

                x_kwargs = {}
                if 'x_mask' in data:
                    x_kwargs['x_mask'] = data['x_mask'].to(self.device)
                
                for key in data:
                    if key.startswith('c_') :
                        if torch.is_tensor(data[key]):
                            x_kwargs[key] = data[key].to(self.device)
                        else:
                            x_kwargs[key] = data[key]

                ## one step optimization
                self.optimizer.zero_grad()

                terms = self.model.compute_losses(x, x_kwargs)
                loss = terms['loss'].mean()
                loss.backward()

                self.optimizer.step()
                self._anneal_lr()
                
                ## log and save
                ## log with loguru, plot with Board
                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    ## log with loguru
                    losses = {key: terms[key].mean().item() for key in terms}

                    logger.info(
                        f"[TRAIN] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | Step: {self.step:7d} | Loss: {losses['loss']:8.5f}"
                    )

                    ## plot with Board
                    write_dict = {'step': self.step, 'train/epoch': epoch}
                    for key in losses:
                        write_dict[f'train/{key}'] = losses[key]
                    Board().write(write_dict)

                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    ## save model
                    self._save()
                
                ## update step and check max steps
                self.step += 1
                if self.step > self.max_steps:
                    return

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ Load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_state_dict = torch.load(path)
    model_state_dict = model.state_dict()

    unchanged_weights = []
    used_weights = []
    for key in model_state_dict:
        ## current state and saved state both on single GPU or both on multi GPUs 
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
            logger.info(f'Load parameter {key} for current model.')
            used_weights.append(key)
        
        ## current state on single GPU and saved state on multi GPUs
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
            logger.info(f'Load parameter module.{key} for current model [Trained on multi GPUs].')
            used_weights.append('module.'+key)
        
        if key not in saved_state_dict and 'module.'+key not in saved_state_dict:
            unchanged_weights.append(key)

    unused_weights = []
    for key in saved_state_dict:
        if key not in used_weights:
            unused_weights.append(key)

    for key in unchanged_weights:
        logger.info(f'Unchanged_weight: {key}')
    
    for key in unused_weights:
        logger.info(f'Unused_weight: {key}')
    
    model.load_state_dict(model_state_dict)
