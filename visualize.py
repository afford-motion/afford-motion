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

from utils.visualize import frame2mp4
from utils.misc import smplx_neutral_model, get_meshes_from_smplx
from utils.visualize import skeleton_to_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smplx_neutral_model = smplx_neutral_model.to(device=device)

from smplkit.constants import SKELETON_CHAIN
kinematic_chain = SKELETON_CHAIN.SMPLH['kinematic_chain']

def render_meshes_to_animation(save_path: str, meshes: List, appendix_meshes: List=None):
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
    camera_pose = np.eye(4)
    camera_pose[0:3, -1] = np.array([0.0, 0.0, 6.5])
    camera_pose = camera_pose @ Q(axis=[1, 0, 0], angle=0).transformation_matrix
    
    ## rendering
    for i in range(len(meshes)):
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
    
    joints = data['joints']
    params = data['params']
    text = data['text']
    index = data['index']
    scene_trans = data['scene_trans']
    scene_mesh = data['scene_mesh']

    ## save path
    base_name = os.path.basename(save_path)
    assert int(base_name) == index
    save_path = (save_path + f'-{text[0:112]}.mp4').replace(' ', '_')

    ## smplx body meshes
    if render_joint:
        body_meshes = skeleton_to_mesh(joints.reshape(-1, 22, 3), kinematic_chain)
    else:
        params_tensor = torch.from_numpy(params).to(device=device).unsqueeze(0)
        verts, faces = get_meshes_from_smplx(smplx_neutral_model, params_tensor)
        verts = verts.squeeze(0).cpu().numpy()

        body_meshes = [trimesh.Trimesh(vertices=verts[i], faces=faces) for i in range(len(verts))]

    ## scene mesh
    scene_mesh = trimesh.load(scene_mesh, process=False)
    scene_mesh.apply_transform(scene_trans)

    # S = trimesh.Scene()
    # S.add_geometry(scene_mesh)
    # S.add_geometry(body_meshes[0])
    # S.add_geometry(trimesh.creation.axis(origin_size=0.06, axis_radius=0.015, axis_length=0.6))
    # S.show()

    ## render
    render_meshes_to_animation(
        save_path,
        body_meshes,
        appendix_meshes=[scene_mesh, trimesh.creation.axis(0.05)],
    )

    return body_meshes, scene_mesh

def save_meshes(meshes, prefix, basename, render_joint=False):
    for i, mesh in enumerate(meshes):
        if render_joint:
            mesh_path = os.path.join(prefix, f'{basename}', f'sk_{i:0>3d}.ply')
        else:
            mesh_path = os.path.join(prefix, f'{basename}', f'bo_{i:0>3d}.obj')
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        mesh.export(mesh_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--cnt', type=int, default=30)
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--save_scene', action='store_true')
    parser.add_argument('--render_joint', action='store_true')
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

        body_meshes, scene_mesh = rendering(f, os.path.join(prefix, 'video', f'{basename}'), args.render_joint)
        if args.save_mesh:
            save_meshes(body_meshes, os.path.join(prefix, 'meshes'), basename, args.render_joint)
        if args.save_scene:
            scene_mesh.export(os.path.join(prefix, 'meshes', basename, 'scene.ply'))
