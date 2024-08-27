import os, sys
sys.path.append(os.path.abspath('.'))
import pickle
import glob
import argparse
import torch
import numpy as np
import smplkit as sk
from tqdm import tqdm
from typing import Tuple
from natsort import natsorted
from pyquaternion import Quaternion as Q

JOINTS = 22 # 22 joints without hands, jaw, eyes

body_model_neutral = sk.SMPLXLayer(gender='neutral', num_pca_comps=-1).to(device='cpu')

def convert_smplx_to_pos_and_aug(smplx: Tuple, same_betas: bool=False) -> np.ndarray:
    """ Convert raw smplx representation to pos representation (for HumanML3D dataset) """
    joints = convert_smplx_to_pos(smplx, same_betas=same_betas)

    # augment
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]

    mjoints = joints.copy().reshape(-1, JOINTS, 3)
    mjoints[:, :, 0] *= -1
    tmp = mjoints[:, right_chain, :]
    mjoints[:, right_chain, :] = mjoints[:, left_chain, :]
    mjoints[:, left_chain, :] = tmp
    mjoints = mjoints.reshape(-1, JOINTS*3).astype(np.float32)

    return joints, mjoints

def convert_smplx_to_pos(smplx: Tuple, same_betas: bool=False) -> np.ndarray:
    """ Convert raw smplx representation to pos representation

    Args:
        smplx: A tuple containing motion sequence and betas
        same_betas: Whether to use the same betas for all poses
    
    Return:
        A pos_rot representation
    """
    pose_seq, betas = smplx
    
    transl = torch.from_numpy(pose_seq[:, :3]).float()
    orient = torch.from_numpy(pose_seq[:, 3:6]).float()
    body_pose = torch.from_numpy(pose_seq[:, 6:69]).float()
    hand_pose = torch.from_numpy(pose_seq[:, 69:]).float()
    betas = torch.from_numpy(betas).float().unsqueeze(0)
    if same_betas:
        betas = betas * 0.0

    joints = body_model_neutral(
        transl=transl,
        orient=orient,
        betas=betas,
        body_pose=body_pose,
        left_hand_pose=hand_pose[:, :45],
        right_hand_pose=hand_pose[:, 45:],
        return_joints=True
    )
    joints = joints.numpy()
    joints = joints[:, :JOINTS, :] # <nframes, JOINTS, 3>

    return joints.reshape(-1, JOINTS*3).astype(np.float32) # <nframes, JOINTS*3>

def smplx_to_vec(smplx: Tuple, dataset: str, save_path: str) -> np.ndarray:
    """ Convert raw smplx representation to specific vector representation

    Args:
        smplx: A tuple containing motion sequence and betas
        dataset: Dataset name
        idn: Motion id
        save_path: Path to save the vector representation

    Return:
        A vector representation
    """
    if dataset == 'HumanML3D':
        vec_repr, aug_vec_repr = convert_smplx_to_pos_and_aug(smplx, same_betas=True)

        ## save vec_repr
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, vec_repr)
        ## save aug vec_repr
        dirname, basename = os.path.dirname(save_path), os.path.basename(save_path)
        basename = 'M' + basename
        np.save(os.path.join(dirname, basename), aug_vec_repr)
    else:
        vec_repr = convert_smplx_to_pos(smplx, same_betas=True)

        ## save vec_repr
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, vec_repr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset') # ['HumanML3D', 'PROX', 'HUMANISE']
    args = parser.parse_args()
    
    datasets = [args.dataset]
    for dataset in datasets:
        ## load smplx
        pkls = natsorted(glob.glob(f"./data/{dataset}/motions/*.pkl"))
        for pkl in tqdm(pkls, desc=f"Processing {dataset}.."):
            save_path = pkl.replace('motions', 'motions_pos').replace('.pkl', '.npy')

            smplx_to_vec(
                pickle.load(open(pkl, 'rb')),
                dataset,
                save_path
            )
        
        print(f"Finished processing {dataset} dataset..")
