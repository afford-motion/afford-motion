import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse
import json
import random
import pickle
import pyrender
import trimesh
import torch
import numpy as np
from PIL import Image
from typing import List
from natsort import natsorted
from pyquaternion import Quaternion as Q
from easydict import EasyDict

from utils.visualize import frame2mp4
from utils.misc import smplx_neutral_model, get_meshes_from_smplx
from utils.visualize import skeleton_to_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smplx_neutral_model = smplx_neutral_model.to(device=device)

SKELETON_CHAIN = EasyDict({
    'SMPL': {
        'kinematic_chain': [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]
    },
    'SMPLH': {
        'kinematic_chain': [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]],
        'left_hand_chain': [[20, 34, 35, 36], [20, 22, 23, 24], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]],
        'right_hand_chain': [[21, 49, 50, 51], [21, 37, 38, 39], [21, 40, 41, 42], [21, 46, 47, 48],[21, 43, 44, 45]],
    },
    'SMPLX': {
        'kinematic_chain': [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15, 22], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20], [15, 23], [15, 24]],
        'left_hand_chain': [[20, 37, 38, 39], [20, 25, 26, 27], [20, 28, 29, 30], [20, 34, 35, 36], [20, 31, 32, 33]],
        'right_hand_chain': [[21, 52, 53, 54], [21, 40, 41, 42], [21, 43, 44, 45], [21, 49, 50, 51], [21, 46, 47, 48]],
    }
})
kinematic_chain = SKELETON_CHAIN.SMPLH['kinematic_chain']

def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def render_meshes_to_animation(save_path: str, meshes: List, appendix_meshes: List=None, z_up: bool=False):
    """ Render meshes to videos

    Args:
        save_dir: directory to save videos
        meshes: meshes to render
        appendix_meshes: appendix meshes to render
    """
    save_img_dir = os.path.join(os.path.dirname(save_path), 'img')
    os.makedirs(save_img_dir, exist_ok=True)

    ## camera
    # H, W = 1080, 1920
    # camera_center = np.array([951.30, 536.77])
    # camera = pyrender.camera.IntrinsicsCamera(
    #     fx=1060.53, fy=1060.38,
    #     cx=camera_center[0], cy=camera_center[1])
    H, W = 1080, 1080
    camera_center = np.array([540.0, 540.0])
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060, fy=1060,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.6)
    
    ## camera pose
    # camera_pose = np.eye(4)
    # camera_pose[0:3, -1] = np.array([0.0, 6.5, 0])
    # camera_pose = camera_pose @ Q(axis=[1, 0, 0], angle=-np.pi / 2).transformation_matrix
    
    ## rendering
    for i in range(len(meshes)):
        angle = np.pi / 6
        R = 3
        if z_up:
            camera_pose = np.eye(4)
            camera_pose[:3, -1] = meshes[i].vertices.mean(axis=0) + np.array([0, -R, np.cos(angle) * R], dtype=np.float32)
            camera_pose = camera_pose @ Q(axis=[1, 0, 0], angle=angle).transformation_matrix
        else:
            ## y up
            camera_pose = np.eye(4)
            camera_pose[:3, -1] = meshes[i].vertices.mean(axis=0) + np.array([0, np.sin(angle) * R, R], dtype=np.float32)
            camera_pose = camera_pose @ Q(axis=[1, 0, 0], angle=-angle).transformation_matrix
        
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)

        body_mesh = pyrender.Mesh.from_trimesh(meshes[i], smooth=False)
        scene.add(body_mesh)
        for m in appendix_meshes:
            am = pyrender.Mesh.from_trimesh(m, smooth=False)
            scene.add(am)
        
        r = pyrender.OffscreenRenderer(
            viewport_width=W,
            viewport_height=H,
        )
        color, _ = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = Image.fromarray((color * 255).astype(np.uint8))
        r.delete()
        save_img_path = os.path.join(save_img_dir, f'{i:03d}.png')
        img.save(save_img_path)

    frame2mp4(os.path.join(save_img_dir, '%03d.png'), save_path)
    os.system(f"rm -rf {save_img_dir}")

def rendering(file_path, save_path, render_joint=False):
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)

    name = data['name']
    text = data['text']
    tokens = data['tokens']
    motion = data['motion']
    m_len = data['m_len']
    motion = motion[0:m_len, :]

    ## save path
    save_path = (save_path + f'-{text[0:112]}.mp4').replace(' ', '_')

    ## smplx body meshes
    pose_seq = motion
    pose_seq = recover_from_ric(torch.from_numpy(pose_seq), 22) # (l, j, 3)
    pose_seq = pose_seq.cpu().numpy()

    body_meshes = skeleton_to_mesh(pose_seq, kinematic_chain, 22)
    
    # S = trimesh.Scene()
    # S.add_geometry(body_meshes[0])
    # S.add_geometry(trimesh.creation.axis(origin_size=0.06, axis_radius=0.015, axis_length=0.6))
    # S.show()

    ## render
    render_meshes_to_animation(
        save_path,
        body_meshes,
        appendix_meshes=[trimesh.creation.axis(origin_size=0.06, axis_radius=0.015, axis_length=0.6)],
    )

    return body_meshes

def save_meshes(meshes, prefix, basename):
    for i, mesh in enumerate(meshes):
        mesh_path = os.path.join(prefix, f'{basename}', f'{i:0>3d}.obj')
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        mesh.export(mesh_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--cnt', type=int, default=30)
    parser.add_argument('--save_mesh', action='store_true')
    args = parser.parse_args()

    files = []
    if args.folder != '' and os.path.exists(args.folder):
        files = natsorted(os.listdir(args.folder))
        random.seed(0)
        random.shuffle(files)
        files = [os.path.join(args.folder, file) for file in files]
    elif args.file != '' and os.path.exists(args.file):
        files = [args.file]
    else:
        raise ValueError('Invalid path or folder')

    for f in files[0:args.cnt]:
        prefix = os.path.dirname(os.path.dirname(f))
        basename = os.path.basename(f).split('.')[0]

        body_meshes = rendering(f, os.path.join(prefix, 'video', f'{basename}'))
        if args.save_mesh:
            save_meshes(body_meshes, os.path.join(prefix, 'meshes'), basename)
