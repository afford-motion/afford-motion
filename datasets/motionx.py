import os
import random
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted

from datasets.base import DATASET
from datasets.transforms import make_default_transform
from utils.misc import compute_repr_dimesion

def full_name(dataset: str, scene_id: str, folder: bool=False) -> str:
    if dataset == 'HUMANISE':
        return f'{scene_id}/{scene_id}_vh_clean_2' if folder == True else f'{scene_id}_vh_clean_2'
    else:
        return f'{scene_id}'

def translation_to_transform(translation: np.ndarray) -> np.ndarray:
    """ Convert translation to transform matrix
    """
    transform = np.eye(4, dtype=np.float32)
    transform[0:3, -1] = translation
    return transform

@DATASET.register()
class MotionXDataset(Dataset):
    """ Cross modal motion dataset
    """
    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        """ Initialize the dataset

        Args:
            cfg: configuration object
            phase: phase string, can be 'train' and 'test'
        """
        self.cfg = cfg
        self.phase = phase
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0

        self.data_dir = cfg.data_dir
        self.sets = cfg.sets
        self.sets_config = cfg.sets_config
        self.shuffle_seed = cfg.shuffle_seed
        
        ## motion configuration
        self.motion_type = cfg.data_repr
        self.contact_joints = cfg.data_repr_joints
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.min_horizon = cfg.min_horizon
        self.max_horizon = cfg.max_horizon

        ## transform configuration
        if self.phase =='train' or self.phase == 'all':
            self.transforms_list = cfg.train_transforms
        else:
            self.transforms_list = cfg.test_transforms
        self.transform = make_default_transform(self.transforms_list, cfg.transform_cfg)

        self._load_datasets()
        self._prepare_statistics()
    
    def _load_split_ids(self):
        split_ids = defaultdict(list)

        for s in self.sets:
            txt = os.path.join(self.data_dir, f'{s}/{self.phase}.txt')
            if s == 'HumanML3D' and not self.sets_config.HumanML3D.use_mirror:
                txt = os.path.join(self.data_dir, f'{s}/{self.phase}_without_mirror.txt')
            with open(txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                split_ids[s].append(int(line))
        
        return split_ids

    def _load_datasets(self):
        """ Load the dataset """
        split_ids = self._load_split_ids()

        self.all_data = []
        for s in self.sets:
            set_data = []
            anno = pd.read_csv(os.path.join(self.data_dir, f'{s}/contact_motion/anno.csv'))
            for i in tqdm(range(len(anno))):
                if i not in split_ids[s]:
                    continue

                scene_id = anno.loc[i]['scene_id']
                scene_id = '' if type(scene_id) != str else scene_id
                scene_trans = np.array([
                    anno.loc[i]['scene_trans_x'],
                    anno.loc[i]['scene_trans_y'],
                    anno.loc[i]['scene_trans_z'],
                ], dtype=np.float32)
                desc = anno.loc[i]['utterance']
                desc = [] if type(desc) != str or desc == '' else desc.split('$$')

                motion = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/motions/{i:0>5}.npy'))
                if motion.shape[0] < self.min_horizon or motion.shape[0] > self.max_horizon:
                    continue
                set_data.append((s, i, scene_id, scene_trans, desc))
            if self.gpu == 0:
                logger.info(f"Load {len(set_data)} cases in {s} dataset")
            self.all_data.extend(set_data)
        
        self.indices = list(range(len(self.all_data)))
        if self.phase == 'train' or self.phase == 'all':
            random.shuffle(self.indices)
        if self.phase == 'test':
            random.seed(self.shuffle_seed) # for test set, we use the same random seed to ensure the same order
            random.shuffle(self.indices)

    def _prepare_statistics(self):
        """ Prepare statistics for normalization """
        mean_std_path = os.path.join(self.data_dir, f"Mean_Std_MotionX_{'_'.join(self.sets)}_{self.motion_type}.npz")
        try:
            npzfile = np.load(mean_std_path)
            self.mean = npzfile['mean']
            self.std = npzfile['std']
            if self.gpu == 0:
                logger.info(f"Load mean and std from {mean_std_path}")
        except:
            all_poses = []
            for s, i, _, _, _ in self.all_data:
                motion = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/motions/{i:0>5}.npy'))
                b, j, d = motion.shape
                all_poses.append(motion.reshape(b, j * d))
            all_poses = np.concatenate(all_poses, axis=0)

            self.mean = all_poses.mean(axis=0, keepdims=True)
            self.std = all_poses.std(axis=0, keepdims=True)
            np.savez(mean_std_path, mean=self.mean, std=self.std)
            if self.gpu == 0:
                logger.info(f"Save mean and std to {mean_std_path}")

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
        return len(self.all_data)

    def __getitem__(self, idx):
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]
        
        s, i, scene_id, scene_trans, desc = self.all_data[index]
        cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts/{i:0>5}.npz')
        if s == 'HumanML3D' and self.sets_config.HumanML3D.use_fur:
            cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts_fur/{i:0>5}.npz')
        contact = np.load(cont_file)
        motion = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/motions/{i:0>5}.npy'))
        points = contact['points'].astype(np.float32)

        ## text
        if len(desc) == 0:
            text = ''
        else:
            text = random.choice(desc)
        
        ## scene
        xyz = points[:, 0:3]
        feat = points[:, 3:3]
        if self.use_color:
            color = (points[:, 3:6] + 1) / 2.0 # [-1, 1] -> [0, 1]
            feat = np.concatenate([feat, color], axis=-1)

        ## motions
        motion = motion.reshape(motion.shape[0], -1)
        l, d = motion.shape
        motion = np.concatenate([
            motion,
            np.zeros((self.max_horizon - l, d), dtype=np.float32)
        ], axis=0)
        motion_mask = np.concatenate([
            np.zeros((l,), dtype=bool),
            np.ones((self.max_horizon - l,), dtype=bool)
        ], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            ## conditions
            'c_text': text,
            'c_pc_xyz': xyz,
            'c_pc_feat': feat,
            ## for visualization or evaluation
            'info_set': s,
            'info_index': i,
            'info_scene_trans': translation_to_transform(scene_trans),
            'info_scene_mesh': os.path.join(self.data_dir, f'{s}/scenes/{full_name(s, scene_id, True)}.ply'),
        }

        if self.phase == 'test':
            if s == 'HUMANISE':
                target_mask = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/target_mask/{i:0>5}.npy'))
                data['info_obj_mask'] = target_mask
            else:
                data['info_obj_mask'] = None

        if self.transform is not None:
            data = self.transform(data)

        ## normalize motion
        data['x'] = self.normalize(data['x'])
        
        return data

@DATASET.register()
class MotionXExampleDataset(MotionXDataset):

    def __init__(self, cfg: DictConfig, phase: str, **kwargs: Dict):
        self.data_path = kwargs['data_path'] if 'data_path' in kwargs else ''
        super().__init__(cfg, phase, **kwargs)
    
    def _load_datasets(self):
        """ Load the dataset """
        ## collect test data
        self.all_data = []

        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip('\n')
                desc, dataset, index, nframes = line.split('#')
                index = int(index)

                anno = pd.read_csv(os.path.join(self.data_dir, f'{dataset}/contact_motion/anno.csv'))
                scene_id = anno.loc[index]['scene_id']
                scene_id = '' if type(scene_id) != str else scene_id
                scene_trans = np.array([
                    anno.loc[index]['scene_trans_x'],
                    anno.loc[index]['scene_trans_y'],
                    anno.loc[index]['scene_trans_z'],
                ], dtype=np.float32)

                motion = np.load(os.path.join(self.data_dir, f'{dataset}/contact_motion/motions/{index:0>5}.npy'))
                if desc == '' or nframes == '':
                    desc = anno.loc[index]['utterance']
                    desc = [] if type(desc) != str or desc == '' else desc.split('$$')
                else:
                    desc, nframes = [desc], int(nframes)
                    motion = np.zeros((int(nframes), *motion.shape[1:]), dtype=np.float32)

                self.all_data.append((dataset, index, scene_id, scene_trans, desc, motion))
        self.indices = None
    
    def __getitem__(self, idx):
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]
        
        s, i, scene_id, scene_trans, desc, motion = self.all_data[index]
        cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts/{i:0>5}.npz')
        if s == 'HumanML3D' and self.sets_config.HumanML3D.use_fur:
            cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts_fur/{i:0>5}.npz')
        contact = np.load(cont_file)
        points = contact['points'].astype(np.float32)

        ## text
        if len(desc) == 0:
            text = ''
        else:
            text = random.choice(desc)
        
        ## scene
        xyz = points[:, 0:3]
        feat = points[:, 3:3]
        if self.use_color:
            color = (points[:, 3:6] + 1) / 2.0 # [-1, 1] -> [0, 1]
            feat = np.concatenate([feat, color], axis=-1)

        ## motions
        motion = motion.reshape(motion.shape[0], -1)
        l, d = motion.shape
        motion = np.concatenate([
            motion,
            np.zeros((self.max_horizon - l, d), dtype=np.float32)
        ], axis=0)
        motion_mask = np.concatenate([
            np.zeros((l,), dtype=bool),
            np.ones((self.max_horizon - l,), dtype=bool)
        ], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            ## conditions
            'c_text': text,
            'c_pc_xyz': xyz,
            'c_pc_feat': feat,
            ## for visualization or evaluation
            'info_set': s,
            'info_index': i,
            'info_scene_trans': translation_to_transform(scene_trans),
            'info_scene_mesh': os.path.join(self.data_dir, f'{s}/scenes/{full_name(s, scene_id, True)}.ply'),
        }

        if self.transform is not None:
            data = self.transform(data)
        
        ## normalize motion
        data['x'] = self.normalize(data['x'])
        
        return data

@DATASET.register()
class MotionXCustomDataset(MotionXDataset):
    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        self.data_path = kwargs['data_path'] if 'data_path' in kwargs else ''
        self.motion_dim = compute_repr_dimesion(cfg.data_repr)
        super().__init__(cfg, phase, **kwargs)

    def _load_datasets(self):
        self.all_data = []
        s = "custom"

        anno = pd.read_csv(os.path.join(self.data_dir, 'custom/anno.csv'))
        for i in tqdm(range(len(anno))):
            scene_id = anno.loc[i]['scene_id']
            scene_id = '' if type(scene_id) != str else scene_id
            scene_trans = np.array([
                anno.loc[i]['scene_trans_x'],
                anno.loc[i]['scene_trans_y'],
                anno.loc[i]['scene_trans_z'],
            ], dtype=np.float32)
            desc = anno.loc[i]['utterance']
            tokens = anno.loc[i]['others']
            nframes = anno.loc[i]['frame']

            scene = np.load(os.path.join(self.data_dir, f'{s}/points/{i:0>4}.npz'))
            points = scene['points'].astype(np.float32)

            self.all_data.append((s, i, scene_id, scene_trans, desc, points, tokens, nframes))
        
        self.indices = list(range(len(self.all_data)))
        assert self.phase == 'test', 'Only support test phase for custom dataset'
        random.seed(self.shuffle_seed)
        random.shuffle(self.indices)
    
    def __getitem__(self, idx: int):
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]

        s, i, scene_id, scene_trans, desc, points, tokens, nframes = self.all_data[index]
        scene_trans = -scene_trans

        ## text
        text = desc

        ## scene
        xyz = points[:, 0:3]
        feat = points[:, 3:3]
        if self.use_color:
            color = (points[:, 3:6]) / 255.0
            feat = np.concatenate([feat, color], axis=-1)
        
        ## motion
        motion = np.zeros((self.max_horizon, self.motion_dim), dtype=np.float32)
        motion_mask = np.concatenate([
            np.zeros((nframes,), dtype=bool),
            np.ones((self.max_horizon - nframes,), dtype=bool)
        ], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            ## conditions
            'c_text': text,
            'c_pc_xyz': xyz,
            'c_pc_feat': feat,
            ## for visualization or evaluation
            'info_tokens': tokens,
            'info_set': s,
            'info_index': i,
            'info_scene_trans': translation_to_transform(scene_trans),
            'info_scene_mesh': os.path.join(self.data_dir, f'{s}/scenes/{scene_id}.ply')
        }

        if self.transform is not None:
            data = self.transform(data)
        
        ## normalize motion
        data['x'] = self.normalize(data['x'])

        return data

@DATASET.register()
class ContactMapDataset(Dataset):
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
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0

        self.data_dir = cfg.data_dir
        self.sets = cfg.sets
        self.sets_config = cfg.sets_config
        self.shuffle_seed = cfg.shuffle_seed
        
        ## contact map config
        self.contact_type = cfg.data_repr
        self.contact_joints = cfg.data_repr_joints
        self.use_raw_dist = cfg.use_raw_dist
        self.sigma = cfg.sigma
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.use_openscene = cfg.use_openscene
        self.point_feat_dim = cfg.point_feat_dim

        ## transform configuration
        if self.phase == 'train' or self.phase == 'all':
            self.transform_list = cfg.train_transforms
        else:
            self.transform_list = cfg.test_transforms
        self.transform = make_default_transform(self.transform_list, cfg.transform_cfg)

        self._load_datasets()
        self._prepare_statistics()

    def _load_split_ids(self):
        split_ids = defaultdict(list)

        for s in self.sets:
            txt = os.path.join(self.data_dir, f'{s}/{self.phase}.txt')
            if s == 'HumanML3D' and not self.sets_config.HumanML3D.use_mirror:
                txt = os.path.join(self.data_dir, f'{s}/{self.phase}_without_mirror.txt')
            with open(txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                split_ids[s].append(int(line))
        
        return split_ids

    def _load_datasets(self):
        """ Load the dataset """
        split_ids = self._load_split_ids()

        self.all_data = []
        for s in self.sets:
            set_data = []
            anno = pd.read_csv(os.path.join(self.data_dir, f'{s}/contact_motion/anno.csv'))
            for i in tqdm(range(len(anno))):
                if i not in split_ids[s]:
                    continue

                scene_id = anno.loc[i]['scene_id']
                scene_id = '' if type(scene_id) != str else scene_id
                scene_trans = np.array([
                    anno.loc[i]['scene_trans_x'],
                    anno.loc[i]['scene_trans_y'],
                    anno.loc[i]['scene_trans_z'],
                ], dtype=np.float32)
                desc = anno.loc[i]['utterance']
                desc = [] if type(desc) != str or desc == '' else desc.split('$$')

                set_data.append((s, i, scene_id, scene_trans, desc))
            if self.gpu == 0:
                logger.info(f"Load {len(set_data)} cases in {s} dataset")
            self.all_data.extend(set_data)
        
        self.indices = list(range(len(self.all_data)))
        if self.phase == 'train' or self.phase == 'all':
            random.shuffle(self.indices)
        if self.phase == 'test':
            random.seed(self.shuffle_seed) # for test set, we use the same random seed to ensure the same order
            random.shuffle(self.indices)

    def _prepare_statistics(self):
        """ Prepare the statistics for normalization """
        if self.use_raw_dist:
            mean_std_path = os.path.join(self.data_dir, f"Mean_Std_Dist_{'_'.join(self.sets)}_{self.contact_type}.npz")
        else:
            mean_std_path = os.path.join(self.data_dir, f"Mean_Std_Cont_{'_'.join(self.sets)}_{self.contact_type}_{self.sigma}.npz")
        if 'HumanML3D' in self.sets and self.sets_config.HumanML3D.use_fur:
            mean_std_path = mean_std_path.replace('.npz', '_fur.npz')
        
        try:
            npzfile = np.load(mean_std_path)
            self.mean = npzfile['mean']
            self.std = npzfile['std']
            if self.gpu == 0:
                logger.info(f"Load mean and std from {mean_std_path}")
        except:
            contact_list = []
            for s, i, _, _, _ in self.all_data:
                cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts/{i:0>5}.npz')
                if s == 'HumanML3D' and self.sets_config.HumanML3D.use_fur:
                    cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts_fur/{i:0>5}.npz')
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
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]
        
        s, i, scene_id, scene_trans, desc = self.all_data[index]
        cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts/{i:0>5}.npz')
        if s == 'HumanML3D' and self.sets_config.HumanML3D.use_fur:
            cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts_fur/{i:0>5}.npz')
        contact = np.load(cont_file)
        points = contact['points'].astype(np.float32)
        dist = contact['dist'].astype(np.float32)

        ## text
        if len(desc) == 0:
            text = ''
        else:
            text = random.choice(desc)
        
        ## scene and contact
        xyz = points[:, 0:3]
        feat = points[:, 3:3]
        if self.use_color:
            color = (points[:, 3:6] + 1) / 2.0 # [-1, 1] -> [0, 1]
            feat = np.concatenate([feat, color], axis=-1)
        if self.use_openscene:
            if self.point_feat_dim == 1 and os.path.exists(
                os.path.join(self.data_dir, f'{s}/contact_motion/affordance/{i:0>5}.npy')):
                openscene = np.load(os.path.join(self.data_dir,
                                f'{s}/contact_motion/affordance/{i:0>5}.npy')).astype(np.float32)
            else:
                mask = contact['mask']
                openscene = np.load(os.path.join(self.data_dir, 
                                f'{s}/feat/{full_name(s, scene_id)}_openscene_feat_distill.npy'))[mask]
            feat = np.concatenate([feat, openscene], axis=-1)
        
        contact = self._extract_contact(dist)
        if not self.use_raw_dist:
            contact = np.exp(-0.5 * contact ** 2 / self.sigma ** 2)
        
        ## prepare data
        data = {
            'x': contact,
            ## conditions
            'c_pc_xyz': xyz,
            'c_pc_feat': feat,
            'c_text': text,
            ## for visualization or saving results
            'info_set': s,
            'info_index': i,
            'info_scene_trans': translation_to_transform(scene_trans),
            'info_scene_mesh': os.path.join(self.data_dir, f'{s}/scenes/{full_name(s, scene_id, True)}.ply'),
        }

        if self.phase == 'test':
            if s == 'HUMANISE':
                target_mask = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/target_mask/{i:0>5}.npy'))
                data['info_obj_mask'] = target_mask
            else:
                data['info_obj_mask'] = None

        if self.transform is not None:
            data = self.transform(data)

        ## normalize contact map
        data['x'] = self.normalize(data['x'])

        return data

@DATASET.register()
class ContactMapExampleDataset(ContactMapDataset):

    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        self.data_path = kwargs['data_path'] if 'data_path' in kwargs else ''
        super().__init__(cfg, phase, **kwargs)
    
    def _load_datasets(self):
        """ Load the dataset """
        self.all_data = []

        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                desc, dataset, index = line.split('#')[0:3]
                desc, index = [desc], int(index)

                anno = pd.read_csv(os.path.join(self.data_dir, f'{dataset}/contact_motion/anno.csv'))
                scene_id = anno.loc[index]['scene_id']
                scene_id = '' if type(scene_id) != str else scene_id
                scene_trans = np.array([
                    anno.loc[index]['scene_trans_x'],
                    anno.loc[index]['scene_trans_y'],
                    anno.loc[index]['scene_trans_z'],
                ], dtype=np.float32)

                self.all_data.append((dataset, index, scene_id, scene_trans, desc))
        self.indices = None
    
    def __len__(self):
        return len(self.all_data)

@DATASET.register()
class ContactMotionDataset(Dataset):
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
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0

        self.data_dir = cfg.data_dir
        self.sets = cfg.sets
        self.sets_config = cfg.sets_config
        self.shuffle_seed = cfg.shuffle_seed
        
        self.motion_type = cfg.data_repr
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
    
    def _load_split_ids(self):
        split_ids = defaultdict(list)

        for s in self.sets:
            txt = os.path.join(self.data_dir, f'{s}/{self.phase}.txt')
            if s == 'HumanML3D' and not self.sets_config.HumanML3D.use_mirror:
                txt = os.path.join(self.data_dir, f'{s}/{self.phase}_without_mirror.txt')
            with open(txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                split_ids[s].append(int(line))
        
        return split_ids
    
    def _load_datasets(self):
        """ Load the dataset """
        split_ids = self._load_split_ids()

        self.all_data = []
        for s in self.sets:
            set_data = []
            anno = pd.read_csv(os.path.join(self.data_dir, f'{s}/contact_motion/anno.csv'))
            for i in tqdm(range(len(anno))):
                if i not in split_ids[s]:
                    continue

                scene_id = anno.loc[i]['scene_id']
                scene_id = '' if type(scene_id) != str else scene_id
                scene_trans = np.array([
                    anno.loc[i]['scene_trans_x'],
                    anno.loc[i]['scene_trans_y'],
                    anno.loc[i]['scene_trans_z'],
                ], dtype=np.float32)
                desc = anno.loc[i]['utterance']
                desc = [] if type(desc) != str or desc == '' else desc.split('$$')

                motion = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/motions/{i:0>5}.npy'))
                if motion.shape[0] < self.min_horizon or motion.shape[0] > self.max_horizon:
                    continue
                set_data.append((s, i, scene_id, scene_trans, desc))
            if self.gpu == 0:
                logger.info(f"Load {len(set_data)} cases in {s} dataset")
            self.all_data.extend(set_data)
        
        self.indices = list(range(len(self.all_data)))
        if self.phase == 'train' or self.phase == 'all':
            random.shuffle(self.indices)
        if self.phase == 'test':
            random.seed(self.shuffle_seed) # for test set, we use the same random seed to ensure the same order
            random.shuffle(self.indices)
    
    def _prepare_statistics(self):
        """ Prepare statistics for normalization """
        mean_std_path = os.path.join(self.data_dir, f"Mean_Std_CM_{'_'.join(self.sets)}_{self.motion_type}.npz")
        try:
            npzfile = np.load(mean_std_path)
            self.mean = npzfile['mean']
            self.std = npzfile['std']
            if self.gpu == 0:
                logger.info(f"Load mean and std from {mean_std_path}")
        except:
            all_poses = []
            for s, i, _, _, _ in self.all_data:
                motion = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/motions/{i:0>5}.npy'))
                b, j, d = motion.shape
                all_poses.append(motion.reshape(b, j * d))
            all_poses = np.concatenate(all_poses, axis=0)

            self.mean = all_poses.mean(axis=0, keepdims=True)
            self.std = all_poses.std(axis=0, keepdims=True)
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
        return len(self.all_data)

    def __getitem__(self, idx: int):
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]
        
        s, i, scene_id, scene_trans, desc = self.all_data[index]
        cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts/{i:0>5}.npz')
        if s == 'HumanML3D' and self.sets_config.HumanML3D.use_fur:
            cont_file = os.path.join(self.data_dir, f'{s}/contact_motion/contacts_fur/{i:0>5}.npz')
        contact = np.load(cont_file)
        motion = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/motions/{i:0>5}.npy'))
        points = contact['points'].astype(np.float32)
        dist = contact['dist'].astype(np.float32)

        ## text
        if len(desc) == 0:
            text = ''
        else:
            text = random.choice(desc)
        
        ## scene and contact
        xyz = points[:, 0:3]
        contact = self._extract_contact(dist)
        if self.phase == 'test':
            ## load the pre-generated dist map
            ## Note that, the pre-generated dist map array contains multi samples, i.e, the contact shape is (k, n, j)
            contact = np.load(os.path.join(self.contact_folder, f'{s}/pred_contact/{i:0>5}.npy')) 
        if self.phase == 'train' or self.phase == 'all':
            ## using pretrained generated dist to train the model
            if np.random.random() < self.mix_train_ratio:
                contact_file = os.path.join(self.data_dir, f"{s}/pred_contact/{i:0>5}.npy")
                if os.path.exists(contact_file):
                    contact = np.load(contact_file).squeeze(0)
        if not self.use_raw_dist:
            contact = np.exp(-0.5 * contact ** 2 / self.sigma ** 2)
        
        ## motion
        motion = motion.reshape(motion.shape[0], -1)
        l, d = motion.shape
        motion = np.concatenate([
            motion,
            np.zeros((self.max_horizon - l, d), dtype=np.float32)
        ], axis=0)
        motion_mask = np.concatenate([
            np.zeros((l,), dtype=bool),
            np.ones((self.max_horizon - l,), dtype=bool)
        ], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            ## conditions
            'c_pc_xyz': xyz,
            'c_pc_contact': contact,
            'c_text': text,
            ## for visualization
            'info_set': s,
            'info_index': i,
            'info_scene_trans': translation_to_transform(scene_trans),
            'info_scene_mesh': os.path.join(self.data_dir, f'{s}/scenes/{full_name(s, scene_id, True)}.ply'),
        }

        if self.phase == 'test':
            if s == 'HUMANISE':
                target_mask = np.load(os.path.join(self.data_dir, f'{s}/contact_motion/target_mask/{i:0>5}.npy'))
                data['info_obj_mask'] = target_mask
            else:
                data['info_obj_mask'] = None

        if self.transform is not None:
            data = self.transform(data)
        
        ## normalize motion
        data['x'] = self.normalize(data['x'])

        return data

@DATASET.register()
class ContactMotionExampleOriginDataset(ContactMotionDataset):

    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        self.data_path = kwargs['data_path'] if 'data_path' in kwargs else ''
        super().__init__(cfg, phase, **kwargs)
    
    def _load_datasets(self):
        """ Load the datasets """
        self.all_data = []

        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip('\n')
                _, dataset, index, _ = line.split('#')
                index = int(index)

                anno = pd.read_csv(os.path.join(self.data_dir, f'{dataset}/contact_motion/anno.csv'))
                scene_id = anno.loc[index]['scene_id']
                scene_id = '' if type(scene_id) != str else scene_id
                scene_trans = np.array([
                    anno.loc[index]['scene_trans_x'],
                    anno.loc[index]['scene_trans_y'],
                    anno.loc[index]['scene_trans_z'],
                ], dtype=np.float32)
                desc = anno.loc[index]['utterance']
                desc = [] if type(desc) != str or desc == '' else desc.split('$$')
                self.all_data.append((dataset, index, scene_id, scene_trans, desc))
        self.indices = None

@DATASET.register()
class ContactMotionExampleDataset(ContactMotionDataset):

    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        self.data_path = kwargs['data_path'] if 'data_path' in kwargs else ''
        self.contact_folder = kwargs['contact_folder'] if 'contact_folder' in kwargs else ''
        super().__init__(cfg, phase, **kwargs)
    
    def _load_datasets(self):
        """ Load the datasets """
        ## pre-generated contacts
        pred_contacts = []
        if self.contact_folder != '':
            contact_files = natsorted(glob.glob(os.path.join(self.contact_folder, '*-*', 'contact.npy')))
            for contact_file in contact_files:
                contact = np.load(contact_file).astype(np.float32)
                pred_contacts.append(contact)
            
            assert len(pred_contacts) > 0, f"Cannot find any predicted contacts in {self.contact_folder}"
            if self.gpu == 0:
                logger.info(f"Load {len(pred_contacts)} predicted contacts")

        ## collect test data
        self.all_data = []

        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip('\n')
                desc, dataset, index, nframes = line.split('#')
                desc, index, nframes = [desc], int(index), int(nframes)

                anno = pd.read_csv(os.path.join(self.data_dir, f'{dataset}/contact_motion/anno.csv'))
                scene_id = anno.loc[index]['scene_id']
                scene_id = '' if type(scene_id) != str else scene_id
                scene_trans = np.array([
                    anno.loc[index]['scene_trans_x'],
                    anno.loc[index]['scene_trans_y'],
                    anno.loc[index]['scene_trans_z'],
                ], dtype=np.float32)

                xyz = pred_contacts[i][:, 0:3]
                dist = pred_contacts[i][:, 3:]
                motion = np.load(os.path.join(self.data_dir, f'{dataset}/contact_motion/motions/{index:0>5}.npy'))
                motion = np.zeros((int(nframes), *motion.shape[1:]), dtype=np.float32)

                self.all_data.append((dataset, index, scene_id, scene_trans, desc, xyz, dist, motion))
        self.indices = None
    
    def __getitem__(self, idx):
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]
        
        s, i, scene_id, scene_trans, desc, xyz, dist, motion = self.all_data[index]
        ## text
        if len(desc) == 0:
            text = ''
        else:
            text = random.choice(desc)
        
        ## scene and contact
        xyz = xyz
        contact = dist
        if not self.use_raw_dist:
            contact = np.exp(-0.5 * contact ** 2 / self.sigma ** 2)
        
        ## motion
        motion = motion.reshape(motion.shape[0], -1)
        l, d = motion.shape
        motion = np.concatenate([
            motion,
            np.zeros((self.max_horizon - l, d), dtype=np.float32)
        ], axis=0)
        motion_mask = np.concatenate([
            np.zeros((l,), dtype=bool),
            np.ones((self.max_horizon - l,), dtype=bool)
        ], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            ## conditions
            'c_pc_xyz': xyz,
            'c_pc_contact': contact,
            'c_text': text,
            ## for visualization
            'info_set': s,
            'info_index': i,
            'info_scene_trans': translation_to_transform(scene_trans),
            'info_scene_mesh': os.path.join(self.data_dir, f'{s}/scenes/{full_name(s, scene_id, True)}.ply'),
        }

        if self.transform is not None:
            data = self.transform(data)
        
        ## normalize motion
        data['x'] = self.normalize(data['x'])

        return data

@DATASET.register()
class ContactMapCustomDataset(ContactMapDataset):
    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        self.data_path = kwargs['data_path'] if 'data_path' in kwargs else ''
        self.contact_dim = compute_repr_dimesion(cfg.data_repr)
        super().__init__(cfg, phase, **kwargs)
    
    def _load_datasets(self):
        """ Load the dataset """
        self.all_data = []
        s = "custom"

        anno = pd.read_csv(os.path.join(self.data_dir, 'custom/anno.csv'))
        for i in tqdm(range(len(anno))):
            scene_id = anno.loc[i]['scene_id']
            scene_id = '' if type(scene_id) != str else scene_id
            scene_trans = np.array([
                anno.loc[i]['scene_trans_x'],
                anno.loc[i]['scene_trans_y'],
                anno.loc[i]['scene_trans_z'],
            ], dtype=np.float32)
            desc = anno.loc[i]['utterance']

            self.all_data.append((s, i, scene_id, scene_trans, desc))
        
        self.indices = list(range(len(self.all_data)))
        assert self.phase == 'test', "Custom dataset only support test phase"
        random.seed(self.shuffle_seed) # for test set, we use the same random seed to ensure the same order
        random.shuffle(self.indices)
    
    def __getitem__(self, idx):
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]
        
        s, i, scene_id, scene_trans, desc = self.all_data[index]
        scene_trans = -scene_trans
        scene = np.load(os.path.join(self.data_dir, f'{s}/points/{i:0>4}.npz'))
        points = scene['points'].astype(np.float32)

        ## text
        text = desc

        ## scene and contact
        xyz = points[:, 0:3]
        feat = points[:, 3:3]
        if self.use_color:
            color = (points[:, 3:6]) / 255.0
            feat = np.concatenate([feat, color], axis=-1)
        
        contact = np.zeros((xyz.shape[0], self.contact_dim), dtype=np.float32)

        ## prepare data
        data = {
            'x': contact,
            ## conditions
            'c_pc_xyz': xyz,
            'c_pc_feat': feat,
            'c_text': text,
            ## for visualization or saving results
            'info_set': s,
            'info_index': i,
            'info_scene_trans': translation_to_transform(scene_trans),
            'info_scene_mesh': os.path.join(self.data_dir, f'{s}/scenes/{scene_id}.ply'),
        }

        if self.transform is not None:
            data = self.transform(data)
        
        ## normalize contact map
        data['x'] = self.normalize(data['x'])

        return data

@DATASET.register()
class ContactMotionCustomDataset(ContactMotionDataset):

    def __init__(self, cfg: DictConfig, phase: str, **kwargs):
        self.data_path = kwargs['data_path'] if 'data_path' in kwargs else ''
        self.motion_dim = compute_repr_dimesion(cfg.data_repr)
        super().__init__(cfg, phase, **kwargs)
    
    def _load_datasets(self):
        """ Load the dataset """
        self.all_data = []
        s = "custom"

        anno = pd.read_csv(os.path.join(self.data_dir, 'custom/anno.csv'))
        for i in tqdm(range(len(anno))):
            scene_id = anno.loc[i]['scene_id']
            scene_id = '' if type(scene_id) != str else scene_id
            scene_trans = np.array([
                anno.loc[i]['scene_trans_x'],
                anno.loc[i]['scene_trans_y'],
                anno.loc[i]['scene_trans_z'],
            ], dtype=np.float32)
            desc = anno.loc[i]['utterance']
            tokens = anno.loc[i]['others']
            nframes = anno.loc[i]['frame']

            scene = np.load(os.path.join(self.data_dir, f'{s}/points/{i:0>4}.npz'))
            points = scene['points'].astype(np.float32)
            xyz = points[:, 0:3]

            self.all_data.append((s, i, scene_id, scene_trans, desc, xyz, tokens, nframes))
        
        self.indices = list(range(len(self.all_data)))
        assert self.phase == 'test', "Custom dataset only support test phase"
        random.seed(self.shuffle_seed) # for test set, we use the same random seed to ensure the same order
        random.shuffle(self.indices)

    def __getitem__(self, idx: int):
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]

        s, i, scene_id, scene_trans, desc, xyz, tokens, nframes = self.all_data[index]
        scene_trans = -scene_trans
        ## text
        text = desc

        ## scene and contact
        xyz = xyz
        contact = None
        if self.phase == 'test':
            ## load the pre-generated dist map
            ## Note that, the pre-generated dist map array contains multi samples, i.e, the contact shape is (k, n, j)
            contact = np.load(os.path.join(self.contact_folder, f'{s}/pred_contact/{i:0>5}.npy'))

        assert contact is not None
        if not self.use_raw_dist:
            contact = np.exp(-0.5 * contact ** 2 / self.sigma ** 2)
        
        ## motion
        motion = np.zeros((self.max_horizon, self.motion_dim), dtype=np.float32)
        motion_mask = np.concatenate([
            np.zeros((nframes,), dtype=bool),
            np.ones((self.max_horizon - nframes,), dtype=bool)
        ], axis=0)

        data = {
            'x': motion,
            'x_mask': motion_mask,
            ## conditions
            'c_pc_xyz': xyz,
            'c_pc_contact': contact,
            'c_text': text,
            ## for visualization
            'info_tokens': tokens,
            'info_set': s,
            'info_index': i,
            'info_scene_trans': translation_to_transform(scene_trans),
            'info_scene_mesh': os.path.join(self.data_dir, f'{s}/scenes/{scene_id}.ply')
        }

        if self.transform is not None:
            data = self.transform(data)
        
        ## normalize motion
        data['x'] = self.normalize(data['x'])

        return data
