import os
import cv2
import torch
import glob
import trimesh
import numpy as np
import pyrender
from PIL import Image
from natsort import natsorted
from typing import Dict, List, Any
from omegaconf import DictConfig
from pyquaternion import Quaternion as Q
from smplkit.constants import SKELETON_CHAIN
from trimesh import transform_points

from utils.registry import Registry

kinematic_chain = SKELETON_CHAIN.SMPLH['kinematic_chain'] # remove the hands, jaw, and eyes

Visualizer = Registry('visualizer')

@Visualizer.register()
class ContactVisualizer():

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.visualizer
        self.data_repr = cfg.dataset.data_repr
        self.data_repr_joints = cfg.dataset.data_repr_joints

        if self.data_repr == 'contact_one_joints' or self.data_repr == 'contact_pelvis':
            self.vis_joints = [0]
        elif self.data_repr == 'contact_all_joints':
            self.vis_joints = cfg.visualizer.vis_joints
        elif self.data_repr == 'contact_cont_joints':
            self.vis_joints = list(range(len(self.data_repr_joints)))
        else:
            raise ValueError(f"Unknown data representation: {self.data_repr} in contact model.")
    
    def visualize(self, sample: torch.Tensor, save_dir: str, *args, **kwargs) -> None:
        """ Visualize samples

        Args:
            samples: samples to visualize
            save_dir: directory to save samples
        """
        ibatch = args[0]
        dataloader = args[1]

        b, n, d = sample.shape
        for i in range(b):
            contact = sample[i].detach().cpu().numpy() # (n, j)
            contact = dataloader.dataset.denormalize(contact, clip=True)
            
            if dataloader.dataset.use_raw_dist:
                dist = contact.copy()
            else:
                dist = np.sqrt(-2 * np.log(contact) * dataloader.dataset.sigma ** 2)
                
            if dataloader.dataset.use_raw_dist:
                contact = contact.clip(0, 2.0) / 2.0
                contact = 1 - contact
            xyz = kwargs['c_pc_xyz'][i].cpu().numpy()
            text = kwargs['c_text'][i]

            for j in self.vis_joints:
                contact_map = contact[:, j:j+1]
                contact_map = np.uint8(255 * contact_map)
                heatmap = cv2.applyColorMap(contact_map, cv2.COLORMAP_PARULA)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)

                save_path= os.path.join(save_dir, f'{ibatch * b + i:03d}-{text}/contact_joint_{j:02d}.ply')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                trimesh.PointCloud(vertices=xyz, colors=heatmap).export(save_path)
            
            scene_contact = np.concatenate([xyz, dist], axis=-1).astype(np.float32)
            np.save(os.path.join(save_dir, f'{ibatch * b + i:03d}-{text}/contact.npy'), scene_contact)

@Visualizer.register()
class ContactMotionVisualizer():

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.visualizer
        self.kinematic_chain = kinematic_chain
    
    def visualize(self, sample: torch.Tensor, save_dir: str, *args, **kwargs) -> None:
        """ Visualize samples
        Args:
            save_dir: directory to save samples
            samples: samples to visualize
        """
        ibatch = args[0]
        dataloader = args[1]

        b, n, d = sample.shape
        for i in range(b):
            text = kwargs['c_text'][i]

            mask = kwargs['x_mask'][i]
            pose_seq = sample[i, ~mask, ...].detach().cpu().numpy()
            pose_seq = dataloader.dataset.denormalize(pose_seq)

            skeleton = pose_seq[:, :self.cfg.njoints * 3]
            skeleton = skeleton.reshape(-1, self.cfg.njoints, 3)
            meshes = skeleton_to_mesh(skeleton, self.kinematic_chain, self.cfg.njoints)

            appendix_meshes = [trimesh.creation.axis(origin_size=0.05)]
            scene_path = kwargs['info_scene_mesh'][i]
            if scene_path is not None and scene_path != '':
                scene_trans = kwargs['info_scene_trans'][i]
                scene_mesh = trimesh.load(scene_path, process=False)
                if len(scene_trans.shape) == 1:
                    scene_mesh.apply_translation(scene_trans)
                else:
                    scene_mesh.apply_transform(scene_trans)
                appendix_meshes.append(scene_mesh)

            render_meshes_to_animation(
                os.path.join(save_dir, f'{ibatch * b + i:03d}-{text}'),
                meshes, appendix_meshes,
                ext='mp4'
            )

@Visualizer.register()
class MotionXVisualizer(ContactMotionVisualizer):

    def visualize_rec(self, x: torch.Tensor, rec_x: torch.Tensor, save_dir: str, *args, **kwargs) -> None:
        """ Visualize reconstruction results

        Args:
            x: grount truth data
            rec_x: reconstruction data
            save_dir: directory to save samples
        """
        ibatch = args[0]
        dataloader = args[1]

        b, n, d = x.shape
        for i in range(b):
            text = kwargs['c_text'][i]
            mask = kwargs['x_mask'][i]

            pose_seq_gt = x[i, ~mask, ...].detach().cpu().numpy()
            pose_seq_gt = dataloader.dataset.denormalize(pose_seq_gt)
            skeleton_gt = pose_seq_gt[:, :self.cfg.njoints * 3]
            skeleton_gt = skeleton_gt.reshape(-1, self.cfg.njoints, 3)
            meshes_gt = skeleton_to_mesh(skeleton_gt, self.kinematic_chain, self.cfg.njoints)

            pose_seq_rec = rec_x[i, ~mask, ...].detach().cpu().numpy()
            pose_seq_rec = dataloader.dataset.denormalize(pose_seq_rec)
            skeleton_rec = pose_seq_rec[:, :self.cfg.njoints * 3]
            skeleton_rec = skeleton_rec.reshape(-1, self.cfg.njoints, 3)
            meshes_rec = skeleton_to_mesh(skeleton_rec, self.kinematic_chain, self.cfg.njoints)

            appendix_meshes = [trimesh.creation.axis(origin_size=0.05)]
            scene_path = kwargs['info_scene_mesh'][i]
            if scene_path is not None and scene_path != '':
                scene_trans = kwargs['info_scene_trans'][i]
                scene_mesh = trimesh.load(scene_path, process=False)
                if len(scene_trans.shape) == 1:
                    scene_mesh.apply_translation(scene_trans)
                else:
                    scene_mesh.apply_transform(scene_trans)
                appendix_meshes.append(scene_mesh)
            
            ## render gt
            render_meshes_to_animation(
                os.path.join(save_dir, f'{ibatch * b + i:03d}-{text}-gt'),
                meshes_gt, appendix_meshes,
                ext='mp4'
            )

            ## render rec
            render_meshes_to_animation(
                os.path.join(save_dir, f'{ibatch * b + i:03d}-{text}-rec'),
                meshes_rec, appendix_meshes,
                ext='mp4'
            )

@Visualizer.register()
class H3DVisualizer():

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.visualizer
        self.kinematic_chain = kinematic_chain
    
    def visualize(self, sample: torch.Tensor, save_dir: str, *args, **kwargs) -> None:
        """ Visualize samples
        Args:
            save_dir: directory to save samples
            samples: samples to visualize
        """
        ibatch = args[0]
        dataloader = args[1]

        b, n, d = sample.shape
        for i in range(b):
            text = kwargs['c_text'][i]

            mask = kwargs['x_mask'][i]
            pose_seq = sample[i, ~mask, ...].detach().cpu().numpy()
            pose_seq = dataloader.dataset.denormalize(pose_seq) # (l, d)
            pose_seq = recover_from_ric(torch.from_numpy(pose_seq), self.cfg.njoints) # (l, j, 3)
            pose_seq = pose_seq.cpu().numpy()

            skeleton = pose_seq[:, :self.cfg.njoints * 3]
            skeleton = skeleton.reshape(-1, self.cfg.njoints, 3)
            meshes = skeleton_to_mesh(skeleton, self.kinematic_chain, self.cfg.njoints)

            appendix_meshes = [trimesh.creation.axis(origin_size=0.05)]
            render_meshes_to_animation(
                os.path.join(save_dir, f'{ibatch * b + i:03d}-{text}'),
                meshes, appendix_meshes,
                ext='mp4',
                z_up=False
            )

def create_visualizer(cfg: DictConfig, *args, **kwargs):
    """ Create visualizer according to the configuration

    Args:
        cfg: configuration dict
    
    Return:
        Visualizer
    """
    return Visualizer.get(cfg.visualizer.name)(cfg, *args, **kwargs)


## Some utils functions
def skeleton_to_mesh(skeleton: np.ndarray, kinematic_chain: List, njoints: int=22) -> List:
    """ Convert skeleton to meshes

    Args:
        skeleton: skeleton array, joints position, <L, njoints, 3>
        kinematic_chain: kinematic chain, can be none
        njoints: joints number
    
    Return:
        Skeleton mesh list
    """
    meshes = []
    if kinematic_chain is None:
        for f in range(skeleton.shape[0]):
            joints = skeleton[f]
            joints_mesh = []
            for i, joint in enumerate(joints):
                joint_mesh = trimesh.creation.uv_sphere(radius=0.02)
                joint_mesh.apply_translation(joint)
                joints_mesh.append(joint_mesh)
            meshes.append(trimesh.util.concatenate(joints_mesh))
    else:
        colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
        meshes = []
        def plot3D(joints, color=None, linewidth=0.01) -> trimesh.Trimesh:
            if color is not None and color.startswith('#'):
                color = trimesh.visual.color.hex_to_rgba(color)
            else:
                color = np.array([128, 0.0, 0.0, 255], dtype=np.uint8)
            
            lines = []
            for i in range(len(joints) - 1):
                line = trimesh.creation.cylinder(
                    radius=linewidth,
                    segment=joints[i:i+2],
                )
                line.visual.vertex_colors = color
                lines.append(line)
            
            return trimesh.util.concatenate(lines)

        for f in range(skeleton.shape[0]):
            joints = skeleton[f]
            joints_mesh = []
            for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
                if i < 5:
                    linewidth = 0.02
                else:
                    linewidth = 0.01
                
                lines = plot3D(joints[chain], color=color, linewidth=linewidth)
                joints_mesh.append(lines)
            
            meshes.append(trimesh.util.concatenate(joints_mesh))
    
    return meshes

def pcd_to_cubes(xyz: np.ndarray) -> List:
    """ Convert point cloud to cubes

    Args:
        xyz: point cloud, <N, 3>
    
    Return:
        Cube mesh list
    """
    cubes = []
    for i in range(xyz.shape[0]):
        cube = trimesh.creation.box(extents=0.02 * np.ones(3))
        cube.apply_translation(xyz[i])
        cubes.append(cube)
    cubes = trimesh.util.concatenate(cubes)
    
    return cubes

def render_mesh_with_contact_map(save_path: str, mesh: trimesh.Trimesh, camera_pose: np.ndarray):
    """ Render mesh with contact map

    Args:
        save_path: save path
        mesh: mesh to render
        camera_pose: camera pose
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W = 1080, 1920
    camera_center = np.array([951.30, 536.77])
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060.53, fy=1060.38,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.6)

    scene = pyrender.Scene()
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    axis_mesh = trimesh.creation.axis(origin_size=0.02)
    scene.add(pyrender.Mesh.from_trimesh(axis_mesh, smooth=False))

    r = pyrender.OffscreenRenderer(
        viewport_width=W,
        viewport_height=H,
    )
    color, _ = r.render(scene)
    color = color.astype(np.float32) / 255.0
    img = Image.fromarray((color * 255).astype(np.uint8))
    r.delete()
    img.save(save_path)
    
def render_meshes_to_animation(save_path: str, meshes: List, appendix_meshes: List=None, ext: str='mp4', z_up: bool=True):
    """ Render meshes to videos

    Args:
        save_dir: directory to save videos
        meshes: meshes to render
        appendix_meshes: appendix meshes to render
    """
    save_img_dir = os.path.join(os.path.dirname(save_path), 'img')
    os.makedirs(save_img_dir, exist_ok=True)

    H, W = 1080, 1920
    camera_center = np.array([951.30, 536.77])
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060.53, fy=1060.38,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.6)

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
    
    save_path = save_path + '.' + ext
    if ext == 'mp4':
        frame2mp4(os.path.join(save_img_dir, '%03d.png'), save_path)
    elif ext == 'gif':
        frame2gif(save_img_dir, save_path, size=0.5)
    os.system(f"rm -rf {save_img_dir}")

def frame2mp4(frames_path: str, mp4: str, start: int=0, fps: int=30):
    """ Convert image frames to video, use ffmpeg to implement the convertion.

    Args:
        frames_path: image path, a string template
        mp4: save path of video result
        size: resize the image into given size, can be tuple or float type
        fps: the fps of video
    """
    cmd = 'ffmpeg -y -framerate {} -start_number {} -i "{}" -pix_fmt yuv420p "{}"'.format(
        fps, start, frames_path, mp4)
    os.system(cmd)

def frame2gif(frames: Any, gif: str, size: Any=None, duration: int=33.33):
    """ Convert image frames to gif, use PIL to implement the convertion.

    Args:
        frames: a image list or a image directory
        gif: save path of gif result
        size: resize the image into given size, can be tuple or float type
        duration: the duration(ms) of images in gif
    """
    if isinstance(frames, list):
        frames = natsorted(frames)
    elif os.path.isdir(frames):
        frames = natsorted(glob.glob(os.path.join(frames, '*.png')))
    else:
        raise Exception('Unsupported input type.')

    images = []
    for f in frames:
        im = Image.open(f)
        if isinstance(size, tuple):
            im = im.resize(size)
        elif isinstance(size, float):
            im = im.resize((int(im.width * size), int(im.height * size)))
        
        images.append(im)

    img, *imgs = images

    os.makedirs(os.path.dirname(gif), exist_ok=True)
    img.save(fp=gif, format='GIF', append_images=imgs,
            save_all=True, duration=duration, loop=0)

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