import torch
import trimesh
import cv2
import numpy as np
from einops import rearrange
from smplkit.constants import SKELETON_CHAIN

from utils.visualize import recover_from_ric
from utils.visualize import skeleton_to_mesh

kinematic_chain = SKELETON_CHAIN.SMPLH['kinematic_chain'] # remove the hands, jaw, and eyes

def debug_motionx_dataloader(dataloader):
    for data in dataloader:
        for key in data:
            if torch.is_tensor(data[key]):
                print(key, data[key].shape)
            elif isinstance(data[key], np.ndarray):
                print(key, data[key].shape)
            else:
                print(key, data[key])
        
        x = data['x']
        x_mask = data['x_mask']
        text = data['c_text']
        pc_xyz = data['c_pc_xyz']
        pc_feat = data['c_pc_feat']

        for i in range(x.shape[0]):
            pose_seq = x[i].cpu().numpy()
            pose_seq = pose_seq[~x_mask[i].cpu().numpy()]
            desc = text[i]
            xyz = pc_xyz[i].cpu().numpy()
            feat = pc_feat[i].cpu().numpy()

            pose_seq = dataloader.dataset.denormalize(pose_seq)
            skeleton = pose_seq[:, :22 * 3].reshape(-1, 22, 3)
            skeleton_mesh = skeleton_to_mesh(skeleton, kinematic_chain)
            skeleton_mesh = [skeleton_mesh[j] for j in range(0, len(skeleton_mesh), 10)]

            print(desc)
            S = trimesh.Scene()
            S.add_geometry(trimesh.creation.axis())
            S.add_geometry(trimesh.PointCloud(vertices=xyz, colors=(feat * 255).astype(np.uint8)))
            S.add_geometry(skeleton_mesh)
            S.show()

            if data['info_scene_mesh'][i] != '':
                S = trimesh.Scene()
                S.add_geometry(trimesh.creation.axis())
                scene_mesh = trimesh.load(data['info_scene_mesh'][i])
                scene_mesh.apply_transform(data['info_scene_trans'][i])
                S.add_geometry(scene_mesh)
                S.add_geometry(skeleton_mesh)
                S.show()
        exit(0)

def debug_contact_map_dataloader(dataloader):
    for data in dataloader:
        for key in data:
            if torch.is_tensor(data[key]):
                print(key, data[key].shape, data[key].dtype)
            elif isinstance(data[key], np.ndarray):
                print(key, data[key].shape, data[key].dtype)
            else:
                print(key, data[key], type(data[key]))

        x = data['x']
        for i in range(x.shape[0]):
            all_contact = x[i].cpu().numpy()
            all_contact = dataloader.dataset.denormalize(all_contact)

            xyz = data['c_pc_xyz'][i].cpu().numpy()
            if data['c_pc_feat'].shape[-1] == 3:
                rgb = (data['c_pc_feat'][i, :, 0:3].cpu().numpy() * 255).astype(np.uint8)
            elif data['c_pc_feat'].shape[-1] == 1:
                score = data['c_pc_feat'][i, :, 0:1].cpu().numpy()
                score = np.clip(score, np.percentile(score, 5), np.percentile(score, 95))
                score = (score - score.min(0, keepdims=True)) / (score.max(0, keepdims=True) - score.min(0, keepdims=True))
                heatmap = np.uint8(255 * score).reshape(-1)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_PARULA)
                rgb = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)
            else:
                rgb = None
            S = trimesh.Scene()
            S.add_geometry(trimesh.creation.axis())
            S.add_geometry(trimesh.PointCloud(vertices=xyz, vertex_colors=rgb))
            S.show()

            if all_contact.shape[1] == 22:
                joints = [0, 10, 11, 20, 21]
            elif all_contact.shape[1] == 6:
                joints = [0, 1, 2, 3, 4, 5]
            elif all_contact.shape[1] == 1:
                joints = [0]
            else:
                raise NotImplementedError

            for j in joints:
                contact = all_contact[:, j:j+1]
                contact_map = np.uint8(255 * contact)
                heatmap = cv2.applyColorMap(contact_map, cv2.COLORMAP_PARULA)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)
                
                S = trimesh.Scene()
                S.add_geometry(trimesh.creation.axis())
                S.add_geometry(trimesh.PointCloud(vertices=xyz, colors=heatmap))
                S.show()
        exit()

def debug_contact_motion_dataloader(dataloader):
    for data in dataloader:
        for key in data:
            if torch.is_tensor(data[key]):
                print(key, data[key].shape, data[key].dtype)
            elif isinstance(data[key], np.ndarray):
                print(key, data[key].shape, data[key].dtype)
            else:
                print(key, data[key], type(data[key]))
    
        x = data['x']
        x_mask = data['x_mask']
        pc_xyz = data['c_pc_xyz']
        pc_contact = data['c_pc_contact']
        for i in range(x.shape[0]):
            pose_seq = x[i].cpu().numpy()
            pose_seq = pose_seq[~x_mask[i].cpu().numpy()]
            xyz = pc_xyz[i].cpu().numpy()
            contact = pc_contact[i].cpu().numpy()

            pose_seq = dataloader.dataset.denormalize(pose_seq)
            skeleton = pose_seq[:, :22 * 3].reshape(-1, 22, 3)
            skeleton_mesh = skeleton_to_mesh(skeleton, kinematic_chain)
            skeleton_mesh = [skeleton_mesh[j] for j in range(0, len(skeleton_mesh), 10)]

            for j in range(contact.shape[1]):
                contact_map = contact[:, j:j+1]
                contact_map = np.uint8(255 * contact_map)
                heatmap = cv2.applyColorMap(contact_map, cv2.COLORMAP_PARULA)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)

                S = trimesh.Scene()
                S.add_geometry(trimesh.creation.axis())
                S.add_geometry(trimesh.PointCloud(vertices=xyz, colors=heatmap))
                S.add_geometry(skeleton_mesh)
                S.show()    

                if data['info_scene_mesh'][i] != '':
                    S = trimesh.Scene()
                    S.add_geometry(trimesh.creation.axis())
                    scene_mesh = trimesh.load(data['info_scene_mesh'][i])
                    scene_mesh.apply_transform(data['info_scene_trans'][i])
                    S.add_geometry(scene_mesh)
                    S.add_geometry(skeleton_mesh)
                    S.show()
        exit(0)

def debug_h3d_dataloader(dataloader):
    for data in dataloader:
        for key in data:
            if torch.is_tensor(data[key]):
                print(key, data[key].shape, data[key].dtype)
            elif isinstance(data[key], np.ndarray):
                print(key, data[key].shape, data[key].dtype)
            else:
                print(key, data[key], type(data[key]))

        x = data['x']
        x_mask = data['x_mask']
        text = data['c_text']

        for i in range(x.shape[0]):
            pose_seq = x[i].cpu().numpy()
            pose_seq = pose_seq[~x_mask[i].cpu().numpy()]
            desc = text[i]

            pose_seq = dataloader.dataset.denormalize(pose_seq)
            skeleton = recover_from_ric(torch.from_numpy(pose_seq), 22)
            skeleton_mesh = skeleton_to_mesh(skeleton, kinematic_chain)
            skeleton_mesh = [skeleton_mesh[j] for j in range(0, len(skeleton_mesh), 10)]

            print(desc)
            S = trimesh.Scene()
            S.add_geometry(trimesh.creation.axis())
            S.add_geometry(skeleton_mesh)
            S.show()
    
    exit()

def debug_contact_map_h3d_dataloader(dataloader):
    for data in dataloader:
        for key in data:
            if torch.is_tensor(data[key]):
                print(key, data[key].shape, data[key].dtype)
            elif isinstance(data[key], np.ndarray):
                print(key, data[key].shape, data[key].dtype)
            else:
                print(key, data[key], type(data[key]))

        x = data['x']
        for i in range(x.shape[0]):
            all_contact = x[i].cpu().numpy()
            all_contact = dataloader.dataset.denormalize(all_contact)

            xyz = data['c_pc_xyz'][i].cpu().numpy()
            S = trimesh.Scene()
            S.add_geometry(trimesh.creation.axis())
            S.add_geometry(trimesh.PointCloud(vertices=xyz))
            S.show()

            if all_contact.shape[1] == 22:
                joints = [0, 10, 11, 20, 21]
            elif all_contact.shape[1] == 6:
                joints = [0, 1, 2, 3, 4, 5]
            elif all_contact.shape[1] == 1:
                joints = [0]
            else:
                raise NotImplementedError

            for j in joints:
                contact = all_contact[:, j:j+1]
                contact_map = np.uint8(255 * contact)
                heatmap = cv2.applyColorMap(contact_map, cv2.COLORMAP_PARULA)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)
                
                S = trimesh.Scene()
                S.add_geometry(trimesh.creation.axis())
                S.add_geometry(trimesh.PointCloud(vertices=xyz, colors=heatmap))
                S.show()
        exit()

def debug_contact_motion_h3d_dataloader(dataloader):
    for data in dataloader:
        for key in data:
            if torch.is_tensor(data[key]):
                print(key, data[key].shape, data[key].dtype)
            elif isinstance(data[key], np.ndarray):
                print(key, data[key].shape, data[key].dtype)
            else:
                print(key, data[key], type(data[key]))
    
        x = data['x']
        x_mask = data['x_mask']
        pc_xyz = data['c_pc_xyz']
        pc_contact = data['c_pc_contact']
        text = data['c_text']
        for i in range(x.shape[0]):
            pose_seq = x[i].cpu().numpy()
            pose_seq = pose_seq[~x_mask[i].cpu().numpy()]
            xyz = pc_xyz[i].cpu().numpy()
            contact = pc_contact[i].cpu().numpy()
            desc = text[i]

            pose_seq = dataloader.dataset.denormalize(pose_seq)
            skeleton = recover_from_ric(torch.from_numpy(pose_seq), 22)
            skeleton_mesh = skeleton_to_mesh(skeleton, kinematic_chain)
            skeleton_mesh = [skeleton_mesh[j] for j in range(0, len(skeleton_mesh), 10)]

            print(desc)
            for j in range(contact.shape[1]):
                contact_map = contact[:, j:j+1]
                contact_map = np.uint8(255 * contact_map)
                heatmap = cv2.applyColorMap(contact_map, cv2.COLORMAP_PARULA)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)

                S = trimesh.Scene()
                S.add_geometry(trimesh.creation.axis())
                S.add_geometry(trimesh.PointCloud(vertices=xyz, colors=heatmap))
                S.add_geometry(skeleton_mesh)
                S.show()    

        exit(0)