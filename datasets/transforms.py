import torch
import numpy as np
from trimesh import transform_points
from typing import Any, Dict, List


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms: Any) -> None:
        self.transforms = transforms

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Any: 
        for t in self.transforms:
            data = t(data, *args, **kwargs)
        return data

class NumpyToTensor(object):
    """ Convert numpy data to torch.Tensor data
    """
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        for key in data.keys():
            if isinstance(data[key], np.ndarray) and 'info' not in key:
                data[key] = torch.tensor(data[key])
        
        return data

class RandomSetLangNull(object):
    """ Randomly set language condition to null
    """
    def __init__(self, **kwargs) -> None:
        self.random_mask_prob = kwargs.get('random_mask_prob', 0.0)

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        if np.random.rand() < self.random_mask_prob:
            data['c_text'] = ''
        return data

class RandomMaskLang(object):
    """ Randomly mask language condition
    """
    def __init__(self, **kwargs) -> None:
        self.random_mask_prob = kwargs.get('random_mask_prob', 0.0)

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        if np.random.rand() < self.random_mask_prob:
            data['c_text_mask'] = np.ones((1,), dtype=bool)
        else:
            data['c_text_mask'] = np.zeros((1,), dtype=bool)
        
        return data

class RandomEraseLang(object):
    """ Randomly erase language condition, set language feature to zero
    """
    def __init__(self, **kwargs) -> None:
        self.random_mask_prob = kwargs.get('random_mask_prob', 0.0)

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        if np.random.rand() < self.random_mask_prob:
            data['c_text_erase'] = np.ones((1,), dtype=bool)
        else:
            data['c_text_erase'] = np.zeros((1,), dtype=bool)
        
        return data

class RandomSetContactNull(object):
    """ Randomly set contact condition to null
    """
    def __init__(self, **kwargs) -> None:
        self.random_mask_prob_pc = kwargs.get('random_mask_prob_pc', 0.0)
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        if np.random.rand() < self.random_mask_prob_pc:
            data['c_pc_xyz'] = data['c_pc_xyz'] * 0
            data['c_pc_contact'] = data['c_pc_contact'] * 0
        return data

class RandomMaskContact(object):
    """ Randomly mask contact condition
    """
    def __init__(self, **kwargs) -> None:
        self.random_mask_prob_pc = kwargs.get('random_mask_prob_pc', 0.0)
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        if np.random.rand() < self.random_mask_prob_pc:
            data['c_pc_mask'] = np.ones((1,), dtype=bool)
        else:
            data['c_pc_mask'] = np.zeros((1,), dtype=bool)
        return data

class RandomEraseContact(object):
    """ Randomly erase contact condition, set contact feature to zero
    """
    
    def __init__(self, **kwargs) -> None:
        self.random_mask_prob_pc = kwargs.get('random_mask_prob_pc', 0.0)
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        if np.random.rand() < self.random_mask_prob_pc:
            data['c_pc_erase'] = np.ones((1,), dtype=bool)
        else:
            data['c_pc_erase'] = np.zeros((1,), dtype=bool)
        return data

class RandomRotation(object):
    """ Randomly rotate scene and motion
    """
    def __init__(self, **kwargs) -> None:
        self.gravity_dim = kwargs.get('gravity_dim', 2)
        self.angle = [0, 0, 0]
        self.angle[self.gravity_dim] = 1
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float32)
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=np.float32)
        trans_mat = np.eye(4, dtype=np.float32)
        trans_mat[0:3, 0:3] = np.dot(R_z, np.dot(R_y, R_x))

        if 'info_aug_trans' in data:
            data['info_aug_trans'] = trans_mat @ data['info_aug_trans']
        else:
            data['info_aug_trans'] = trans_mat
        
        return data

class NormalizeToCenter(object):
    """ Normalize data to center
    """
    def __init__(self, **kwargs) -> None:
        self.gravity_dim = kwargs.get('gravity_dim', 2)
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        xyz = data['c_pc_xyz']
        xy_center = (xyz[:, 0:2].max(axis=0) + xyz[:, 0:2].min(axis=0)) * 0.5
        z_height = np.percentile(xyz[:, 2], 5) # 5% height
        trans_mat = np.eye(4, dtype=np.float32)
        trans_mat[0:3, -1] -= [xy_center[0], xy_center[1], z_height]

        if 'info_aug_trans' in data:
            data['info_aug_trans'] = trans_mat @ data['info_aug_trans']
        else:
            data['info_aug_trans'] = trans_mat

        return data

class ApplyTransformCDM(object):
    """ Apply transform to data
    """
    def __init__(self, **kwargs) -> None:
        pass
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        data['c_pc_xyz'] = transform_points(data['c_pc_xyz'], data['info_aug_trans']).astype(np.float32)
        data['info_scene_trans'] = data['info_aug_trans'] @ data['info_scene_trans']

        return data

class ApplyTransformCMDM(object):
    """ Apply transform to data
    """
    def __init__(self, **kwargs) -> None:
        pass
    
    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        data['c_pc_xyz'] = transform_points(data['c_pc_xyz'], data['info_aug_trans']).astype(np.float32)

        n, c = data['x'].shape
        motion = data['x'].reshape(-1, 3)
        motion = transform_points(motion, data['info_aug_trans']).astype(np.float32)
        data['x'] = motion.reshape(n, c)

        data['info_scene_trans'] = data['info_aug_trans'] @ data['info_scene_trans']
        return data

TRANSFORMS = {
    'NumpyToTensor': NumpyToTensor,
    'RandomSetLangNull': RandomSetLangNull,
    'RandomMaskLang': RandomMaskLang,
    'RandomEraseLang': RandomEraseLang,
    'RandomSetContactNull': RandomSetContactNull,
    'RandomMaskContact': RandomMaskContact,
    'RandomEraseContact': RandomEraseContact,
    'NormalizeToCenter': NormalizeToCenter,
    'RandomRotation': RandomRotation,
    'ApplyTransformCDM': ApplyTransformCDM,
    'ApplyTransformCMDM': ApplyTransformCMDM,
}

def make_default_transform(transforms_list: List, transform_cfg) -> Compose:
    """ Make default transform

    Args:
        transforms_list: the list of specified transforms
        transform_cfg: transform configuration
    
    Return:
        Composed transforms.
    """

    ## compose transforms
    transforms = []
    for t in transforms_list:
        transforms.append(TRANSFORMS[t](**transform_cfg))

    return Compose(transforms)