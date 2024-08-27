import os, sys
sys.path.append(os.path.abspath('.'))
import glob
import random
import json
import argparse
import trimesh
import csv
import pandas as pd
import numpy as np
import cv2
from typing import Dict, List, Tuple
from natsort import natsorted
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from trimesh import transform_points
from pyquaternion import Quaternion as Q
from smplkit.constants import SKELETON_CHAIN

from utils.visualize import skeleton_to_mesh

kinematic_chain = SKELETON_CHAIN.SMPLH['kinematic_chain'] # remove the hands, jaw, and eyes

semantic_to_label = {
  "wall": 1,
  "floor": 2,
  "cabinet": 3,
  "bed": 4,
  "chair": 5,
  "sofa": 6,
  "table": 7,
  "door": 8,
  "window": 9,
  "bookshelf": 10,
  "picture": 11,
  "counter": 12,
  "blinds": 13,
  "desk": 14,
  "shelves": 15,
  "curtain": 16,
  "dresser": 17,
  "pillow": 18,
  "mirror": 19,
  "floor mat": 20,
  "clothes": 21,
  "ceiling": 22,
  "books": 23,
  "refrigerator": 24,
  "television": 25,
  "paper": 26,
  "towel": 27,
  "shower curtain": 28,
  "box": 29,
  "whiteboard": 30,
  "person": 31,
  "nightstand": 32,
  "toilet": 33,
  "sink": 34,
  "lamp": 35,
  "bathtub": 36,
  "bag": 37,
  "otherstructure": 38,
  "otherfurniture": 39,
  "otherprop": 40
}
label_to_semantic = {v: k for k, v in semantic_to_label.items()}
FLOOR_COLORS = np.array(json.load(open('./prepare/floor_colors.json', 'r'))['floor_colors'], dtype=np.uint8)

def load_humanml3d(min_horizon: int, max_horizon: int, **kwargs: Dict) -> Tuple:
    """ Process humanml3d dataset

    Args:
        min_horizon: minimum horizon for motion sequence
        max_horizon: maximum horizon for motion sequence
    
    Return:
        motions: list of motion-condition pairs, each contains (pose_seq, [text], (scene_id, scene_trans), other_info)
        scene_data: additionally loaded scene data for saving memory, {} will be returned if no scene in this dataset
    """
    FRAMERATE=20
    SCENE_ID = kwargs['floor_id'] # 'floor or random_floor
    
    motion_paths = natsorted(glob.glob('./data/HumanML3D/motions_pos/*.npy'))
    
    motions = []
    scene_data = {}
    for motion_path in tqdm(motion_paths, desc='Loading HumanML3D..'):
        pose_seq = np.load(motion_path)

        if len(pose_seq) < min_horizon or len(pose_seq) > 200:
            continue

        text_path = motion_path.replace('motions_pos', 'texts').replace('.npy', '.txt')
        with open(text_path, 'r') as f:
            texts = []
            texts_ann = []
            flag = False
            for line in f.readlines():
                line_split = line.rstrip('\n').split('#')
                caption = line_split[0]
                cap_ann = line_split[1]

                s_tag = float(line_split[2])
                e_tag = float(line_split[3])
                s_tag = 0.0 if np.isnan(s_tag) else s_tag
                e_tag = 0.0 if np.isnan(e_tag) else e_tag

                if s_tag == 0.0 and e_tag == 0.0:
                    flag = True
                    texts.append(caption)
                    texts_ann.append(cap_ann)
                else:
                    try:
                        ## refer to HumanML3D code, https://github.com/EricGuo5513/text-to-motion
                        ## motion sequences segmented with different length are treated new motions
                        n_pose_seq = pose_seq[int(s_tag*FRAMERATE):int(e_tag*FRAMERATE)]

                        if len(n_pose_seq) < min_horizon or len(n_pose_seq) > max_horizon:
                            raise Exception('Invalid motion sequence length!')

                        ## construct motion-condition pair
                        m_c = (n_pose_seq, [caption], (SCENE_ID, np.eye(4, dtype=np.float32)), {'texts_ann': cap_ann}) # [motion sequence, text list, (scene id, scene trans), other_info]
                        motions.append(m_c)
                    except:
                        pass
            
            if flag == True:
                ## construct motion-condition pair
                if len(pose_seq) > max_horizon:
                    pose_seq = pose_seq[:max_horizon]

                m_c = (pose_seq, texts, (SCENE_ID, np.eye(4, dtype=np.float32)), {'texts_ann': '$$'.join(texts_ann)})
                motions.append(m_c)
    
        if SCENE_ID not in scene_data:
            scene_pcd = np.load(f'./data/HumanML3D/points/{SCENE_ID}.npy') # <x,y,z,r,g,b>
            scene_data[SCENE_ID] = {
                'pcd': scene_pcd.astype(np.float32),
                'mesh_path': f'./data/HumanML3D/scenes/{SCENE_ID}.ply'
            }
        
    return motions, scene_data

def load_humanise(min_horizon: int, max_horizon: int, **kwargs: Dict) -> Tuple:
    """ Load humanise dataset

    Args:
        min_horizon: minimum horizon for motion sequence
        max_horizon: maximum horizon for motion sequence
    
    Return:
        motions: list of motion-condition pairs, each contains (pose_seq, [text], (scene_id, scene_trans), other_info)
        scene_data: additionally loaded scene data for saving memory, {} will be returned if no scene in this dataset
    """
    def resynthesize_description(action: str, object_label: int):
        """ Open-vocabulary? """
        action_str = {
            'sit': 'sit on',
            'walk': 'walk to',
            'stand up': 'stand up from',
            'lie': 'lie down on'
        }[action]
        object_str = f"the {label_to_semantic[object_label]}"
        return f"{action_str} {object_str}"

    FRAMERATE=30

    motion_paths = natsorted(glob.glob('./data/HUMANISE/motions_pos/*.npy'))
    anno_file = pd.read_csv('./data/HUMANISE/annotations.csv')
    assert anno_file.shape[0] == len(motion_paths)
    
    motions = []
    scene_data = {}
    for motion_path in tqdm(motion_paths, desc='Loading HUMANISE..'):
        pose_seq = np.load(motion_path)
        index = int(motion_path.split('/')[-1].split('.')[0])

        if len(pose_seq) < min_horizon or len(pose_seq) > max_horizon:
            continue
        
        texts = [resynthesize_description(
            anno_file.loc[index]['action'], anno_file.loc[index]['object_semantic_label'])]

        scene_id = anno_file.loc[index]['scene_id']
        scene_trans = np.eye(4, dtype=np.float32)
        scene_trans[0:3, -1] = np.array([
            float(anno_file.loc[index]['scene_trans_x']),
            float(anno_file.loc[index]['scene_trans_y']),
            float(anno_file.loc[index]['scene_trans_z']),
        ], dtype=np.float32)

        ## construct motion-condition pair
        m_c = (pose_seq, texts, (scene_id, scene_trans), {'origin_desc': anno_file.loc[index]['text']})
        motions.append(m_c)

        ## load scene point cloud
        if scene_id not in scene_data:
            scene_pcd = np.load(f'./data/HUMANISE/points/{scene_id}.npy') # <x,y,z,r,g,b>
            scene_data[scene_id] = {
                'pcd': scene_pcd.astype(np.float32),
                'mesh_path': f'./data/HUMANISE/scenes/{scene_id}/{scene_id}_vh_clean_2.ply'
            }
    
    return motions, scene_data

def load_prox(min_horizon: int, max_horizon: int, segment_horizon: int, segment_stride: int=1, **kwargs: Dict) -> Tuple:
    """ Load prox dataset

    Args:
        min_horizon: minimum horizon for motion sequence
        max_horizon: maximum horizon for motion sequence
        segment_horizon: fix horizon for segmenting motion sequence
        segment_stride: stride for segmenting motion sequence
    
    Return:
        motions: list of motion-condition pairs, each contains (pose_seq, [text], (scene_id, scene_trans), other_info)
        scene_data: additionally loaded scene data for saving memory, {} will be returned if no scene in this dataset
    """
    FRAMERATE=30

    assert segment_horizon >= min_horizon, 'segment horizon is smaller than min horizon!'
    assert segment_horizon <= max_horizon, 'segment horizon is greater than max horizon!'

    random_segment = kwargs.get('random_segment', False)
    random_segment_window = kwargs.get('random_segment_window', 0)
    assert random_segment_window > 0, 'random segment window must be greater than 0!'

    motion_paths = natsorted(glob.glob('./data/PROX/motions_pos/*.npy'))
    scene_trans = json.load(open('./data/PROX/normalize_to_center.json', 'r'))
    scene_trans = {s: np.array(scene_trans[s], dtype=np.float32) for s in scene_trans}

    motions = []
    scene_data = {}
    for motion_path in tqdm(motion_paths, desc='Loading PROX..'):
        pose_seq = np.load(motion_path)
        scene_id, _, _ = motion_path.split('/')[-1].split('.')[0].split('_')

        for i in range(0, len(pose_seq) - segment_horizon + 1, segment_stride):
            strat_index = i
            if random_segment:
                h = random.randint(segment_horizon-random_segment_window, segment_horizon+random_segment_window)
            else:
                h = segment_horizon
            end_index = min(strat_index + h, len(pose_seq))

            n_pose_seq = pose_seq[strat_index:end_index]
            ## construct motion-condition pair
            m_c = (n_pose_seq, None, (scene_id, scene_trans[scene_id]), {})
            motions.append(m_c)

        ## load scene point cloud
        if scene_id not in scene_data:
            scene_pcd = np.load(f'./data/PROX/points/{scene_id}.npy') # <x,y,z,r,g,b>
            scene_data[scene_id] = {
                'pcd': scene_pcd.astype(np.float32),
                'mesh_path': f'./data/PROX/scenes/{scene_id}.ply'
            }
    
    return motions, scene_data

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Reference: https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default l2
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        return min_y_to_x
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        return min_x_to_y
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]

        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        return min_y_to_x, min_x_to_y
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

def visualize_contact_map(xyz, rgb, contact, joint_id, threshold=0.8, gray=True) -> None:
    """ Visualize contact map on scene point cloud
    
    Args:
        xyz: scene point cloud xyz
        rgb: scene point cloud rgb
        contact: contact map
        joint_id: joint id
        threshold: threshold for contact map
    """
    contact = np.exp(-0.5 * contact ** 2 / 0.5 ** 2)
    color = ((rgb.copy() + 1.0) * 127.5).astype(np.uint8)
    if gray:
        color = color.reshape(-1, 1, 3)
        color = cv2.cvtColor(np.uint8(color), cv2.COLOR_RGB2GRAY).repeat(3, axis=-1)
    
    contact = contact[:, joint_id:joint_id+1]
    overlay_mask = (contact > threshold).reshape(-1)
    contact = contact[overlay_mask]
    if len(contact) != 0:
        contact_map = (contact - contact.min()) / (contact.max() - contact.min())
        contact_map = np.uint8(255 * contact_map)
        heatmap = cv2.applyColorMap(contact_map, cv2.COLORMAP_PARULA)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)
        color[overlay_mask, :] = heatmap
    
    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis())
    S.add_geometry(trimesh.PointCloud(vertices=xyz, colors=color))
    S.show()

def visualize_partial_scene(partial_scene, pose_seq, scene_trans, scene_path):
    """ Visualize partial scene and skeleton
    
    Args:
        partial_scene: partial scene point cloud
        pose_seq: pose sequence
        scene_trans: transformation matrix for scene mesh
        scene_path: scene mesh path
    """

    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis())
    xyz = partial_scene[:, 0:3]
    color = ((partial_scene[:, 3:6] + 1) * 127.5).astype(np.uint8)
    S.add_geometry(trimesh.PointCloud(vertices=xyz, vertex_colors=color))
    skeleton = pose_seq[:, :22 * 3].reshape(-1, 22, 3)
    mesh = skeleton_to_mesh(skeleton, kinematic_chain)
    S.add_geometry(mesh)
    S.show()

    S = trimesh.Scene()
    S.add_geometry(trimesh.creation.axis())
    if scene_path is not None:
        scene_mesh = trimesh.load(scene_path, process=False)
        scene_mesh.apply_translation(scene_trans)
        S.add_geometry(scene_mesh)
    S.add_geometry(mesh)
    S.show()

def process(motions, scene_data, save_dir, num_points: int=8192, region_size: float=4.0, **kwargs) -> None:
    """ Process motion-condition pairs
    
    Args:
        motions: motion data
        scene_data: scene data
        num_points: number of points for each scene point cloud chunk
    """
    JOINTS = 22
    REGION_SIZE = region_size
    TRAJ_PAD = REGION_SIZE * kwargs.get('traj_pad_ratio', 0.5)

    anno_list = []
    for i in tqdm(range(len(motions))):
        pose_seq, texts, (scene_id, scene_trans), other_info = motions[i]
        
        ## pose sequence
        pose_seq = pose_seq.copy().astype(np.float32)
        pelvis_seq = pose_seq[:, :3]
        pose_seq = pose_seq[:, :JOINTS * 3].reshape(-1, JOINTS, 3)
        
        ## text
        if texts is not None:
            utterances = '$$'.join(texts)
        else:
            utterances = ''

        append_infos = ''
        for key in other_info:
            append_infos += other_info[key]
        
        ## scene
        assert scene_id is not None
        scene_trans = scene_trans.copy()[0:3, -1]

        traj_max = pelvis_seq.max(axis=0)[0:2]
        traj_min = pelvis_seq.min(axis=0)[0:2]
        traj_size = traj_max - traj_min
        traj_size = traj_size + TRAJ_PAD * np.exp(-traj_size)

        pad = (REGION_SIZE - traj_size) / 2
        pad = np.maximum(pad, [0, 0])

        center = (traj_max + traj_min) / 2
        center_region_max = center + pad
        center_region_min = center - pad
        sample_xy = np.random.uniform(low=center_region_min, high=center_region_max)
        sample_region_max = sample_xy + REGION_SIZE / 2
        sample_region_min = sample_xy - REGION_SIZE / 2

        scene_pcd = scene_data[scene_id]['pcd'].copy()
        scene_pcd[:, 0:3] += scene_trans
        point_in_region = (scene_pcd[:, 0] >= sample_region_min[0]) & (scene_pcd[:, 0] <= sample_region_max[0]) & \
                            (scene_pcd[:, 1] >= sample_region_min[1]) & (scene_pcd[:, 1] <= sample_region_max[1])
        
        indices = np.arange(len(scene_pcd))[point_in_region]
        assert len(indices) > 0, "No points in the region!"
        if len(indices) < num_points:
            if len(indices) < num_points // 4:
                print(f"Warning: only {len(indices)} points in the region! Less than {num_points // 4} points!")
            while len(indices) < num_points:
                indices = np.concatenate([indices, indices])    
        indices = np.random.choice(indices, num_points, replace=False)

        ## save the partial scene without transformation
        points = scene_data[scene_id]['pcd'].copy()
        points[:, 0:3] += scene_trans
        points = points[indices]

        ## transform the partial scene and motion to center
        xyz = points[:, 0:3]
        xy_center = (xyz[:, 0:2].max(axis=0) + xyz[:, 0:2].min(axis=0)) * 0.5
        z_height = np.percentile(xyz[:, 2], 2) # 2% height
        trans_vec = np.array([-xy_center[0], -xy_center[1], -z_height])
        points[:, 0:3] += trans_vec

        pose_seq += trans_vec

        scene_trans = scene_trans + trans_vec

        ## use the partial scene for computing distance map
        partial_scene = points.copy()
        ## visualize partial scene
        ## for debug
        # print(partial_scene.shape, points.shape, points.dtype, indices.shape, indices.dtype)
        # visualize_partial_scene(partial_scene, pose_seq, scene_trans, scene_data[scene_id]['mesh_path'] if scene_id is not None else None)

        ## dist map
        dist = []
        for j in range(JOINTS):
            joint = pose_seq[:, j, :]
            scene_xyz = partial_scene[:, 0:3]
            c_d = chamfer_distance(joint, scene_xyz, metric='l2', direction='y_to_x')
            dist.append(c_d)
        dist = np.concatenate(dist, axis=-1).astype(np.float32)
        
        ## visualize contact map
        ## for debug
        # visualize_contact_map(partial_scene[:, 0:3], partial_scene[:, 3:6], dist, 10)
        # visualize_contact_map(partial_scene[:, 0:3], partial_scene[:, 3:6], dist, 11)
        # visualize_contact_map(partial_scene[:, 0:3], partial_scene[:, 3:6], dist, 20)
        # visualize_contact_map(partial_scene[:, 0:3], partial_scene[:, 3:6], dist, 21)
        
        ## re-index and save
        save_motion_path = os.path.join(save_dir, 'motions', f'{i:0>5}.npy')
        save_scene_path = os.path.join(save_dir, 'contacts', f'{i:0>5}.npz')
        os.makedirs(os.path.dirname(save_motion_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_scene_path), exist_ok=True)
        with open(save_motion_path, 'wb') as fp:
            np.save(fp, pose_seq)
        with open(save_scene_path, 'wb') as fp:
            np.savez(fp, points=points, mask=indices, dist=dist)

        ## save annotation
        anno_list.append([
            scene_id,
            f"{scene_trans[0]:.8f}",
            f"{scene_trans[1]:.8f}",
            f"{scene_trans[2]:.8f}",
            utterances,
            append_infos
        ])
        
    with open(os.path.join(save_dir, 'anno.csv'), 'w') as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(['scene_id', 'scene_trans_x', 'scene_trans_y', 'scene_trans_z', 'utterance', 'others'])
        csvwriter.writerows(anno_list)

def visualize(motions, scene_data):
    for i in range(10):
        index = random.randint(0, len(motions))
        pose_seq, texts, (scene_id, scene_trans), other_info = motions[index]
        scene_pcd = scene_data[scene_id]['pcd']
        scene_mesh_path = scene_data[scene_id]['mesh_path']
        scene_mesh = trimesh.load(scene_mesh_path, process=False)
        scene_mesh.apply_transform(scene_trans)
        assert len(scene_mesh.vertices) == len(scene_pcd) 

        ## visualize
        S = trimesh.Scene()
        S.add_geometry(trimesh.creation.axis())
        xyz = transform_points(scene_pcd[:, :3], scene_trans)
        color = ((scene_pcd[:, 3:6] + 1) * 127.5).astype(np.uint8)
        S.add_geometry(trimesh.PointCloud(vertices=xyz, vertex_colors=color))

        skeleton = pose_seq[:, :22 * 3].reshape(-1, 22, 3)
        mesh = skeleton_to_mesh(skeleton, kinematic_chain)
        S.add_geometry(mesh)
        S.show()

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_horizon', type=int, default=24)
    parser.add_argument('--max_horizon', type=int, default=196)
    parser.add_argument('--num_points', type=int, default=8192)
    parser.add_argument('--segment_horizon', type=int, default=120)
    parser.add_argument('--segment_stride', type=int, default=4)
    parser.add_argument('--random_segment', action='store_true', default=False)
    parser.add_argument('--random_segment_window', type=int, default=60)
    parser.add_argument('--region_size', type=float, default=4.0)
    parser.add_argument('--traj_pad_ratio', type=float, default=0.5)
    parser.add_argument('--floor_id', type=str, default='random_floor')
    args = parser.parse_args()
    
    DATASET = ['HumanML3D', 'HUMANISE', 'PROX']
    for s in DATASET:
        save_dir = f'./data/{s}/contact_motion/'
        motions, scene_data = eval(f'load_{s.lower()}')(**vars(args))
        # visualize(motions, scene_data)
        process(motions, scene_data, save_dir, **vars(args))
