import os, glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from datasets.base import DATASET
from datasets.transforms import make_default_transform
from utils.misc import compute_repr_dimesion

@DATASET.register()
class HumanML3DDataset(Dataset):
    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        """ Initialize the dataset

        Args:
            cfg: configuration object
            phase: phase string, can be 'train' and 'test'
        """
        self.cfg = cfg
        self.phase = phase
        self.data_dir = cfg.data_dir
        self.shuffle_seed = cfg.shuffle_seed

        self.unit_length = 4

        ## motion configuration
        self.motion_type = cfg.data_repr
        self.motion_dim = compute_repr_dimesion(self.motion_type)
        self.ratio = cfg.ratio
        self.min_horizon = cfg.min_horizon
        self.max_horizon = cfg.max_horizon

        if self.phase == 'train' or self.phase == 'all':
            self.transforms_list = cfg.train_transforms
        else:
            self.transforms_list = cfg.test_transforms
        self.transform = make_default_transform(self.transforms_list, cfg.transform_cfg)

        self._load_datasets()
        self._prepare_statistics()
    
    def _load_datasets(self):
        """ Load the original humanml3d dataset """
        data_dict = {}
        id_list = []
        
        split_file = os.path.join(self.data_dir, 'H3D', f'{self.phase}.txt')
        with open(split_file, 'r') as f:
            for line in f.readlines():
                if random.random() > self.ratio:
                    continue
                id_list.append(line.strip())
        logger.info(f"Load {len(id_list)} cases in H3D, including: {id_list[:100]}...")
        
        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(os.path.join(self.data_dir, 'H3D/new_joint_vecs', name + '.npy'))
                if np.isnan(motion).any():
                    continue
                if (len(motion)) < self.min_horizon or (len(motion) >= 200):
                    continue
                
                text_data = []
                flag = False
                with open(os.path.join(self.data_dir, 'H3D/texts', name + '.txt')) as f:
                    for i, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_dict['caption_idx'] = i
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < self.min_horizon or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

        self.indices = list(range(len(self.data_dict)))
        if self.phase == 'train' or self.phase == 'all':
            random.shuffle(self.indices)
        if self.phase == 'test':
            random.seed(self.shuffle_seed - 2023) # for test set, we use the same random seed to ensure the same order
            random.shuffle(self.indices)

    def _prepare_statistics(self):
        """ Prepare the statistics of the dataset """
        self.mean = np.load(os.path.join(self.data_dir, 'H3D/Mean.npy'))
        self.std = np.load(os.path.join(self.data_dir, 'H3D/Std.npy'))
    
    def get_dataloader(self, **kwargs):
        """ Get dataloader
        """
        return DataLoader(self, **kwargs)

    def normalize(self, pose_seq: np.ndarray) -> np.ndarray:
        """ Normalize pose sequence
        
        Args:
            pose_seq: a numpy array of pose sequence
        
        Return:
            Normalized pose sequence
        """
        return (pose_seq - self.mean) / self.std
    
    def denormalize(self, pose_seq: np.ndarray) -> np.ndarray:
        """ Denormalize pose sequence (usually for visualization)

        Args:
            pose_seq: a numpy array of normalized pose sequence
        
        Return:
            Denormalized pose sequence
        """
        return pose_seq * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx: int):
        index = self.indices[idx]

        name = self.name_list[index]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        assert motion.shape[0] == m_length

        ## text
        if self.phase == 'test':
            text_data = text_list[0] # in test phase, fix the test description
        else:
            text_data = random.choice(text_list)
        caption, tokens, caption_idx = text_data['caption'], text_data['tokens'], text_data['caption_idx']

        ## motions
        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        motion = self.normalize(motion)
        motion = np.concatenate([motion, np.zeros((self.max_horizon - m_length, motion.shape[1]))], axis=0).astype(np.float32)
        motion_mask = np.concatenate([np.zeros(m_length, dtype=bool), np.ones(self.max_horizon - m_length, dtype=bool)], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            'c_text': caption,
            'info_tokens': tokens,
            'info_index': name.split('_')[-1],
            'info_caption_index': caption_idx
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

@DATASET.register()
class HumanML3DExampleDataset(HumanML3DDataset):
    def __init__(self, cfg: DictConfig, phase: str, **kwargs: Dict):
        self.data_path = kwargs['data_path'] if 'data_path' in kwargs else ''
        super().__init__(cfg, phase, **kwargs)
    
    def _load_datasets(self):
        id_list = []
        desc_list = []
        length_list = []
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                idx, desc, length = line.split('#')
                id_list.append(idx)
                desc_list.append(desc)
                length_list.append(int(length) if length != '' else 0)

        data_dict = {}
        for i, name in enumerate(id_list):
            try:
                items = []
                
                motion = np.load(os.path.join(self.data_dir, 'H3D/new_joint_vecs', name + '.npy'))
                if np.isnan(motion).any():
                    data_dict[name] = None
                    continue
                if (len(motion)) < self.min_horizon or (len(motion) >= 200):
                    data_dict[name] = None
                    continue
                
                with open(os.path.join(self.data_dir, 'H3D/texts', name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            items.append({'motion': motion,
                                       'length': len(motion),
                                       'text': text_dict})
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < self.min_horizon or (len(n_motion) >= 200):
                                    continue
                                items.append({'motion': n_motion,
                                            'length': len(n_motion),
                                            'text': text_dict})
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                
                if len(items) == 0:
                    data_dict[name] = None
                else:
                    data_dict[name] = random.choice(items)
            except:
                pass
        
        self.data_dict = data_dict
        self.name_list = id_list
        self.desc_list = desc_list
        self.length_list = length_list
    
    def __getitem__(self, index):
        name = self.name_list[index]
        data = self.data_dict[name]

        length = self.length_list[index]
        desc = self.desc_list[index]
        if length != 0 and desc != '':
            motion, m_length, text = np.zeros((length, self.motion_dim)), length, {'caption': desc, 'tokens': ''}
        else:
            assert data is not None, f'data is None, index: {index}'
            motion, m_length, text = data['motion'], data['length'], data['text']
        
        ## text
        caption, tokens = text['caption'], text['tokens']

        ## motion
        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        motion = self.normalize(motion)
        motion = np.concatenate([motion, np.zeros((self.max_horizon - m_length, motion.shape[1]))], axis=0).astype(np.float32)
        motion_mask = np.concatenate([np.zeros(m_length, dtype=bool), np.ones(self.max_horizon - m_length, dtype=bool)], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            'c_text': caption,
            'info_tokens': tokens,
            'info_index': name.split('_')[-1],
        }

        if self.transform is not None:
            data = self.transform(data)
        
        return data

@DATASET.register()
class ContactHumanML3DDataset(Dataset):
    """ Contact map dataset
    """
    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        """ Initialize the dataset

        Args:
            cfg: configuration object
            phase: phase string, can be 'train' and 'test'
        """
        self.cfg = cfg
        self.phase = phase
        self.data_dir = cfg.data_dir
        self.shuffle_seed = cfg.shuffle_seed
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0
        
        ## contact map config
        self.contact_type = cfg.data_repr
        self.contact_joints = cfg.data_repr_joints
        self.use_raw_dist = cfg.use_raw_dist
        self.sigma = cfg.sigma
        self.num_points = cfg.num_points
        self.min_horizon = cfg.min_horizon
        self.max_horizon = cfg.max_horizon

        ## transform configuration
        if self.phase == 'train' or self.phase == 'all':
            self.transform_list = cfg.train_transforms
        else:
            self.transform_list = cfg.test_transforms
        self.transform = make_default_transform(self.transform_list, cfg.transform_cfg)

        self._load_datasets()
        self._prepare_statistics()

    def _load_datasets(self):
        """ Load the dataset """
        data_dict = {}
        id_list = []

        split_file = os.path.join(self.data_dir, 'H3D', f'{self.phase}.txt')
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        logger.info(f"Load {len(id_list)} cases in H3D")

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(os.path.join(self.data_dir, 'H3D/new_joint_vecs', name + '.npy'))
                if np.isnan(motion).any():
                    continue
                if (len(motion)) < self.min_horizon or (len(motion) >= 200):
                    continue
                
                text_data = []
                flag = False
                with open(os.path.join(self.data_dir, 'H3D/texts', name + '.txt')) as f:
                    for i, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_dict['caption_idx'] = i
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < self.min_horizon or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

        self.indices = list(range(len(self.data_dict)))
        if self.phase == 'train' or self.phase == 'all':
            random.shuffle(self.indices)
        if self.phase == 'test':
            random.seed(self.shuffle_seed - 2023) # for test set, we use the same random seed to ensure the same order
            random.shuffle(self.indices)

    def _prepare_statistics(self):
        """ Prepare the statistics for normalization """
        if self.use_raw_dist:
            mean_std_path = os.path.join(self.data_dir, f"Mean_Std_Dist_OriH3D_{self.contact_type}.npz")
        else:
            mean_std_path = os.path.join(self.data_dir, f"Mean_Std_Cont_OriH3D_{self.contact_type}_{self.sigma}.npz")
        
        try:
            npzfile = np.load(mean_std_path)
            self.mean = npzfile['mean']
            self.std = npzfile['std']
            if self.gpu == 0:
                logger.info(f"Load mean and std from {mean_std_path}")
        except:
            id_list = []
            split_file = os.path.join(self.data_dir, 'H3D/all.txt')
            with open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())
            
            contact_list = []
            for name in id_list:
                cont_file = os.path.join(self.data_dir, f'H3D/contacts/{name}.npz')
                if not os.path.exists(cont_file):
                    continue
                contact = np.load(cont_file)['dist'].astype(np.float32)
                contact = self._extract_contact(contact)
                if not self.use_raw_dist:
                    contact = np.exp(-0.5 * contact ** 2 / self.sigma ** 2)
                contact_list.append(contact)
            contact_list = np.concatenate(contact_list, axis=0)

            self.mean = contact_list.mean(axis=0, keepdims=True)
            self.std = contact_list.std(axis=0, keepdims=True)
            np.savez(mean_std_path, mean=self.mean, std=self.std)
            if self.gpu == 0:
                logger.info(f"Save mean and std to {mean_std_path}")
    
    def _extract_contact(self, contact: np.ndarray) -> np.ndarray:
        """ Extract contact from contact array with different contact types """
        if self.contact_type == 'contact_one_joints':
            contact = contact.max(axis=-1, keepdims=True)
        elif self.contact_type == 'contact_all_joints':
            contact = contact
        elif self.contact_type == 'contact_cont_joints':
            contact = contact[:, self.contact_joints]
        elif self.contact_type == 'contact_pelvis':
            contact = contact[:, [0]]
        else:
            raise ValueError(f"Unknown contact type: {self.contact_type}")
        return contact
    
    def get_dataloader(self, **kwargs):
        """ Get dataloader
        """
        return DataLoader(self, **kwargs)

    def normalize(self, contact: np.ndarray) -> np.ndarray:
        """ Normalize the contact map

        Args:
            contact: the contact map to be normalized

        Returns:
            normalized contact map
        """
        return (contact - self.mean) / self.std
    
    def denormalize(self, contact: np.ndarray, clip: bool=False) -> np.ndarray:
        """ Denormalize the contact map

        Args:
            contact: the contact map to be denormalized
            clip: whether to clip the contact map to [0, 1]

        Returns:
            denormalized contact map
        """
        contact = contact * self.std + self.mean
        if clip:
            if self.use_raw_dist:
                contact = contact.clip(0., )
            else:
                contact = contact.clip(1e-20, 1.)
        
        return contact
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]

        name = self.name_list[index]
        data = self.data_dict[name]
        _, _, text_list = data['motion'], data['length'], data['text']

        ## get original name
        name = name.split('_')[-1]
        cont_file = os.path.join(self.data_dir, f'H3D/contacts/{name}.npz')
        contact = np.load(cont_file)
        points = contact['points'].astype(np.float32)
        dist = contact['dist'].astype(np.float32)

        ## text
        if self.phase == 'test':
            text_data = text_list[0] # in test phase, fix the test description
        else:
            text_data = random.choice(text_list)
        caption, tokens, caption_idx = text_data['caption'], text_data['tokens'], text_data['caption_idx']
        
        ## contact
        xyz = points[:, 0:3]
        contact = self._extract_contact(dist)
        if not self.use_raw_dist:
            contact = np.exp(-0.5 * contact ** 2 / self.sigma ** 2)
        contact = self.normalize(contact)
        
        ## prepare data
        data = {
            'x': contact,
            ## conditions
            'c_pc_xyz': xyz,
            'c_text': caption,
            'info_index': name,
            'info_caption_index': caption_idx
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

@DATASET.register()
class ContactMotionHumanML3DDataset(Dataset):
    """ Contact motion dataset
    """
    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        """ Initialize the dataset

        Args:
            cfg: configuration object
            phase: phase string, can be 'train' and test
        """
        self.cfg = cfg
        self.phase = phase
        self.data_dir = cfg.data_dir
        self.shuffle_seed = cfg.shuffle_seed
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0
        
        self.unit_length = 4

        ## motion configuration
        self.motion_type = cfg.data_repr
        self.motion_dim = compute_repr_dimesion(self.motion_type)
        self.contact_type = cfg.contact_type
        self.contact_joints = cfg.contact_joints
        self.use_raw_dist = cfg.use_raw_dist
        self.sigma = cfg.sigma
        self.num_points = cfg.num_points
        self.max_horizon = cfg.max_horizon
        self.min_horizon = cfg.min_horizon
        self.mix_train_ratio = cfg.mix_train_ratio

        if self.phase == 'test':
            self.contact_folder = kwargs['contact_folder'] # use predict contact map for evaluation
            assert self.contact_folder != '', "Please specify the pre-generated contact folder for testing"

        ## transform configuration
        if self.phase == 'train' or self.phase == 'all':
            self.transform_list = cfg.train_transforms
        else:
            self.transform_list = cfg.test_transforms
        self.transform = make_default_transform(self.transform_list, cfg.transform_cfg)

        self._load_datasets()
        self._prepare_statistics()
    
    def _load_datasets(self):
        """ Load the original humanml3d dataset """
        data_dict = {}
        id_list = []
        
        split_file = os.path.join(self.data_dir, 'H3D', f'{self.phase}.txt')
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        logger.info(f"Load {len(id_list)} cases in H3D")
        
        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(os.path.join(self.data_dir, 'H3D/new_joint_vecs', name + '.npy'))
                if np.isnan(motion).any():
                    continue
                if (len(motion)) < self.min_horizon or (len(motion) >= 200):
                    continue
                
                text_data = []
                flag = False
                with open(os.path.join(self.data_dir, 'H3D/texts', name + '.txt')) as f:
                    for i, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_dict['caption_idx'] = i
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < self.min_horizon or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

        self.indices = list(range(len(self.name_list)))
        if self.phase == 'train' or self.phase == 'all':
            random.shuffle(self.indices)
            ## collect pre-generated contact files
            if self.mix_train_ratio > 0:
                self.pred_contact_dict = defaultdict(list)

                contact_files = glob.glob(os.path.join(self.data_dir, f"H3D/pred_contact/*-*.npy"))
                for f in contact_files:
                    name = os.path.basename(f).split('-')[0]
                    self.pred_contact_dict[name].append(f)
        if self.phase == 'test':
            random.seed(self.shuffle_seed - 2023) # for test set, we use the same random seed to ensure the same order
            random.shuffle(self.indices)
    
    def _prepare_statistics(self):
        """ Prepare the statistics of the dataset """
        self.mean = np.load(os.path.join(self.data_dir, 'H3D/Mean.npy'))
        self.std = np.load(os.path.join(self.data_dir, 'H3D/Std.npy'))
    
    def _extract_contact(self, contact: np.ndarray) -> np.ndarray:
        """ Extract contact from contact array with different contact types """
        if self.contact_type == 'contact_one_joints':
            contact = contact.max(axis=-1, keepdims=True)
        elif self.contact_type == 'contact_all_joints':
            contact = contact
        elif self.contact_type == 'contact_cont_joints':
            contact = contact[:, self.contact_joints]
        elif self.contact_type == 'contact_pelvis':
            contact = contact[:, [0]]
        else:
            raise ValueError(f"Unknown contact type: {self.contact_type}")
        return contact

    def get_dataloader(self, **kwargs):
        """ Get dataloader
        """
        return DataLoader(self, **kwargs)
    
    def normalize(self, motion: np.ndarray) -> np.ndarray:
        """ Normalize the motion

        Args:
            motion: motion data to be normalized
        
        Return:
            Normalized motion
        """
        return (motion - self.mean) / self.std

    def denormalize(self, motion: np.ndarray) -> np.ndarray:
        """ Denormalize the motion

        Args:
            motion: motion data to be denormalized
        
        Return:
            Denormalized motion
        """
        return motion * self.std + self.mean
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        index = self.indices[idx]
        
        name = self.name_list[index]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        assert motion.shape[0] == m_length

        ## text
        if self.phase == 'test':
            text_data = text_list[0] # in test phase, fix the test description
        else:
            text_data = random.choice(text_list)
        caption, tokens, caption_idx = text_data['caption'], text_data['tokens'], text_data['caption_idx']

        ## contact
        cont_file = os.path.join(self.data_dir, f"H3D/contacts/{name.split('_')[-1]}.npz")
        contact = np.load(cont_file)
        points = contact['points'].astype(np.float32)
        dist = contact['dist'].astype(np.float32)

        xyz = points[:, 0:3]
        contact = self._extract_contact(dist)
        if self.phase == 'test':
            ## load the pre-generated dist map
            ## Note that, the pre-generated dist map array contains multi samples, i.e, the contact shape is (k, n, j)
            contact = np.load(os.path.join(self.contact_folder, f"H3D/pred_contact/{name.split('_')[-1]}-{caption_idx}.npy"))
        if self.phase == 'train' or self.phase == 'all':
            ## using pretrained generated dist to train the model
            if np.random.random() < self.mix_train_ratio:
                ori_name = name.split('_')[-1]
                if ori_name in self.pred_contact_dict and len(self.pred_contact_dict[ori_name]) != 0:
                    contact = np.load(np.random.choice(self.pred_contact_dict[ori_name])).squeeze(0)
        if not self.use_raw_dist:
            contact = np.exp(-0.5 * contact ** 2 / self.sigma ** 2)
        
        ## motion
        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        motion = self.normalize(motion)
        motion = np.concatenate([motion, np.zeros((self.max_horizon - m_length, motion.shape[1]))], axis=0).astype(np.float32)
        motion_mask = np.concatenate([np.zeros(m_length, dtype=bool), np.ones(self.max_horizon - m_length, dtype=bool)], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            ## conditions
            'c_pc_xyz': xyz,
            'c_pc_contact': contact,
            'c_text': caption,
            ## for visualization
            'info_tokens': tokens,
            'info_index': name.split('_')[-1],
            'info_caption_index': caption_idx
        }

        if self.transform is not None:
            data = self.transform(data)

        return data
