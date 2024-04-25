import os
import torch
import cv2
import trimesh
import pickle
import json
import numpy as np
from collections import defaultdict
from typing import List
from sklearn.metrics import pairwise_distances
from omegaconf import DictConfig

from utils.registry import Registry
from utils.joints_to_smplx import JointsToSMPLX
from utils.misc import smplx_neutral_model, get_meshes_from_smplx
from utils.eval.eval_humanml import eval_humanml

Evaluator = Registry('evaluator')

class Eval():
    def __init__(self, *args, **kwargs) -> None:
        pass

    def evaluate(self, *args, **kwargs) -> None:
        pass

    def report(self, *args, **kwargs) -> None:
        pass

@Evaluator.register()
class ContactHumanML3DEvaluator(Eval):

    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        self.cfg = cfg.evaluator
        
        self.k_samples = self.cfg.k_samples
        self.num_k_samples = self.cfg.num_k_samples
        self.eval_nbatch = self.cfg.eval_nbatch
        self.save_results = self.cfg.save_results
    
    def evaluate(self, sample_list: List, k_samples_list: List, save_dir: str, dataloader: torch.utils.data.DataLoader, **kwargs):
        """ Evaluate sample

        Args:
            sample_list: sample result
            k_samples_list: k samples result
            save_dir: save directory
            dataloader: dataloader
        """ 
        ## save results
        if self.save_results:
            for i in range(len(sample_list)):
                sample = sample_list[i]

                contact = sample['sample']
                contact = dataloader.dataset.denormalize(contact, clip=True)
                if dataloader.dataset.use_raw_dist:
                    dist = contact.copy()
                else:
                    dist = np.sqrt(-2 * np.log(contact) * dataloader.dataset.sigma ** 2)
                
                name = sample['info_index']
                caption_index = sample['info_caption_index']
                save_path = os.path.join(save_dir, f'H3D/pred_contact/{name}-{caption_index}.npy')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, dist[None, ...])
                
            for i in range(len(k_samples_list)):
                k_samples = k_samples_list[i]

                contact = k_samples['k_samples']
                contact = dataloader.dataset.denormalize(contact, clip=True)
                if dataloader.dataset.use_raw_dist:
                    dist = contact.copy()
                else:
                    dist = np.sqrt(-2 * np.log(contact) * dataloader.dataset.sigma ** 2)

                name = k_samples['info_index']
                caption_index = k_samples['info_caption_index']
                save_path = os.path.join(save_dir, f'H3D/pred_contact/{name}-{caption_index}.npy')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, dist)

@Evaluator.register()
class Text2MotionInSceneHumanML3DEvaluator(Eval):

    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        self.cfg = cfg.evaluator
        self.njoints = self.cfg.njoints

        self.k_samples = self.cfg.k_samples
        self.num_k_samples = self.cfg.num_k_samples
        self.eval_nbatch = self.cfg.eval_nbatch
        self.save_results = self.cfg.save_results
    
    def evaluate(self, sample_list: List, k_samples_list: List, save_dir: str, dataloader: torch.utils.data.DataLoader, **kwargs):
        """ Evaluate sample

        Args:
            sample_list: sample result
            k_samples_list: k samples result
            save_dir: save directory
            dataloader: dataloader
        """ 
        ## save results
        if self.save_results:
            for i in range(len(sample_list)):
                sample = sample_list[i]

                motion = sample['sample']
                ## T2M/MDM first denormalize the generate motion, then normlize it with `mean_for_eval`, as 
                ## https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/motion_loaders/comp_v6_model_dataset.py#L247-L250
                ## We save the denormalized motion here, the normalization will be done in customized evaluation code.
                motion = dataloader.dataset.denormalize(motion)
                length = (~sample['x_mask']).sum()

                name = sample['info_index']
                caption_index = sample['info_caption_index']
                save_path = os.path.join(save_dir, f'humanml/{name}-{caption_index}.pkl')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as fp:
                    pickle.dump({'name': name, 'text': sample['c_text'], 'tokens': sample['info_tokens'], 'motion': motion, 'm_len': length}, fp)
            
            for i in range(len(k_samples_list)):
                k_samples = k_samples_list[i]

                motion = k_samples['k_samples']
                ## But when they evaluate the Multimodality, they directly use the generated motion, without denormalization and then normalization.
                ## See, https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/motion_loaders/model_motion_loaders.py#L14-L46
                length = (~k_samples['x_mask']).sum()

                name = k_samples['info_index']
                caption_index = k_samples['info_caption_index']
                save_path = os.path.join(save_dir, f'humanml/{name}-{caption_index}.pkl')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as fp:
                    pickle.dump({'name': name, 'text': k_samples['c_text'], 'tokens': k_samples['info_tokens'], 'motion': motion, 'm_len': length}, fp)

@Evaluator.register()
class ContactEvaluator(Eval):

    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        self.cfg = cfg.evaluator
        
        self.k_samples = self.cfg.k_samples
        self.num_k_samples = self.cfg.num_k_samples
        self.eval_nbatch = self.cfg.eval_nbatch
        self.eval_metrics = self.cfg.eval_metrics
        self.save_results = self.cfg.save_results

        self.dist_to_target_thresholds = self.cfg.dist_to_target_thresholds

        ## metrics
        self.metrics = defaultdict(list)
    
    def evaluate(self, sample_list: List, k_samples_list: List, save_dir: str, dataloader: torch.utils.data.DataLoader, **kwargs):
        """ Evaluate sample

        Args:
            sample_list: sample result
            k_samples_list: k samples result
            save_dir: save directory
            dataloader: dataloader
        """
        for i in range(len(sample_list)):
            sample = sample_list[i]

            contact = sample['sample']
            contact = dataloader.dataset.denormalize(contact, clip=True)
            if dataloader.dataset.use_raw_dist:
                dist = contact.copy()
            else:
                dist = np.sqrt(-2 * np.log(contact) * dataloader.dataset.sigma ** 2)
            
            if 'dist_to_target' in self.eval_metrics:
                obj_mask = sample['info_obj_mask']
                obj_dist = dist[obj_mask, :]

                dist_to_target = obj_dist.min()
                for threshold in self.dist_to_target_thresholds:
                    if dist_to_target < threshold:
                        self.metrics[f'dist_to_target_{threshold}'].append(1.0)
                    else:
                        self.metrics[f'dist_to_target_{threshold}'].append(0.0)
                self.metrics['dist_to_target_average'].append(obj_dist.mean())
                self.metrics['dist_to_target_pelvis_average'].append(obj_dist[:, 0].mean())
                self.metrics['dist_to_target_min_average'].append(obj_dist.min(-1).mean())
            
        ## save results
        if self.save_results:
            for i in range(len(sample_list)):
                sample = sample_list[i]

                contact = sample['sample']
                contact = dataloader.dataset.denormalize(contact, clip=True)
                if dataloader.dataset.use_raw_dist:
                    dist = contact.copy()
                else:
                    dist = np.sqrt(-2 * np.log(contact) * dataloader.dataset.sigma ** 2)
                
                d_set = sample['info_set']
                index = sample['info_index']
                save_path = os.path.join(save_dir, f'{d_set}/pred_contact/{index:0>5}.npy')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, dist[None, ...])

            for i in range(len(k_samples_list)):
                k_samples = k_samples_list[i]

                contact = k_samples['k_samples']
                contact = dataloader.dataset.denormalize(contact, clip=True)
                if dataloader.dataset.use_raw_dist:
                    dist = contact.copy()
                else:
                    dist = np.sqrt(-2 * np.log(contact) * dataloader.dataset.sigma ** 2)

                d_set = k_samples['info_set']
                index = k_samples['info_index']
                save_path = os.path.join(save_dir, f'{d_set}/pred_contact/{index:0>5}.npy')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, dist)
    
    def report(self, save_dir: str):
        """ Report evaluation results

        Args:
            save_dir: save directory
        """
        save_path = os.path.join(save_dir, f'metrics.txt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            for m in self.metrics:
                f.write(f'{m}: {np.mean(self.metrics[m]):.6f}\n')

@Evaluator.register()
class Text2MotionInSceneEvaluator(Eval):

    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        self.cfg = cfg.evaluator
        self.njoints = self.cfg.njoints

        self.k_samples = self.cfg.k_samples
        self.num_k_samples = self.cfg.num_k_samples
        self.eval_nbatch = self.cfg.eval_nbatch
        self.eval_metrics = self.cfg.eval_metrics
        self.save_results = self.cfg.save_results

        self.opt_rate = self.cfg.opt_rate
        self.opt_steps = self.cfg.opt_steps
        self.joints_to_smplx_model = JointsToSMPLX(opt_rate=self.opt_rate, opt_steps=self.opt_steps).to(self.device)
        self.joints_to_smplx_model.load_and_freeze(self.cfg.joints_to_smplx_model_weights)
        self.smplx_neutral_model = smplx_neutral_model.to(self.device)

        ## metrics
        self.metrics = defaultdict(list)
    
    def joints_to_smplx_mesh(self, joints, joints_mask):
        """ Convert joints to SMPL-X mesh verts

        Args:
            joints: joints position sequence (b, l, j * 3)
            joints_mask: sequence mask (b, l)
        
        Return:
            SMPL-X mesh vertices list [<l, v, 3>, ...]
        """
        params = self.joints_to_smplx_model.joints_to_params_batch(
            self.smplx_neutral_model, joints, joints_mask, optimize=True)
        
        verts_list = []
        for i in range(len(params)):
            verts, _ = get_meshes_from_smplx(self.smplx_neutral_model, params[i].unsqueeze(dim=0))
            verts_list.append(verts.squeeze(dim=0))
        
        return verts_list
    
    def evaluate(self, sample_list: List, k_samples_list: List, save_dir: str, dataloader: torch.utils.data.DataLoader, **kwargs):
        """ Evaluate sample

        Args:
            sample_list: sample result
            k_samples_list: k samples result
            save_dir: save directory
            dataloader: dataloader
        """
        device = kwargs['device'] if 'device' in kwargs else 'cpu'

        for i in range(len(sample_list)):
            item = sample_list[i]
            self.metrics['_name'].append(f"{item['info_set']} - {item['info_index']}")
            self.metrics['_length'].append(f"{(~item['x_mask']).sum()}")

        if 'dist' in self.eval_metrics or 'non_collision' in self.eval_metrics or 'contact' in self.eval_metrics:
            self.eval_physics(sample_list, dataloader, device=device)

        if 'apd' in self.eval_metrics and len(k_samples_list) > 0:
            self.eval_apd(k_samples_list, dataloader, device=device)
        
        if 'Rprecison' in self.eval_metrics or 'fid' in self.eval_metrics:
            self.eval_humanml(sample_list, k_samples_list, dataloader, device=device)
            
        ## save results
        if self.save_results:
            for i in range(len(sample_list)):
                sample = sample_list[i]

                index= sample['info_index']
                save_path = os.path.join(save_dir, f'joints/{index:0>5}.pkl')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                save_joints = sample['sample'] # (l, j * 3)
                save_joints = dataloader.dataset.denormalize(save_joints)
                save_joints = torch.from_numpy(save_joints).float().to(device).unsqueeze(0) # (1, l, j * 3)
                x_mask = sample['x_mask']
                x_mask = torch.from_numpy(x_mask).bool().to(device).unsqueeze(0) # (1, l)

                save_params = self.joints_to_smplx_model.joints_to_params_batch(
                    self.smplx_neutral_model, save_joints, x_mask, optimize=True)
                save_params = save_params[0]

                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'joints': save_joints[~x_mask].cpu().numpy(),
                        'params': save_params.cpu().numpy(),
                        'text': sample['c_text'],
                        'set': sample['info_set'],
                        'index': sample['info_index'],
                        'scene_trans': sample['info_scene_trans'],
                        'scene_mesh': sample['info_scene_mesh'],
                    }, f)

    def eval_physics(self, sample: List, dataloader: torch.utils.data.DataLoader, device='cpu'):
        """ Evaluate physics, including non-collision and contact and dist
        """
        for i in range(len(sample)):
            motion = sample[i]['sample']
            joints = dataloader.dataset.denormalize(motion) # (l, j * 3)
            joints = torch.from_numpy(joints).float().to(device).unsqueeze(0) # (1, l, j * 3)

            ## convert to SMPL-X mesh using projection and optimization
            x_mask = torch.from_numpy(sample[i]['x_mask']).bool().to(device).unsqueeze(0) # (1, l)
            body_verts_list = self.joints_to_smplx_mesh(joints, x_mask)
            body_verts = body_verts_list[0] # <l, v, 3>
            body_faces = self.smplx_neutral_model.faces # <f, 3>
            body_faces = torch.from_numpy(body_faces).long().to(device) # <f, 3>

            xyz = torch.from_numpy(sample[i]['c_pc_xyz']).float().to(device) # <vS, 3>
            assert body_verts.shape[0] == (~x_mask).sum()

            ## compute physics
            if 'non_collision' in self.eval_metrics or 'contact' in self.eval_metrics:
                scene_xyz = xyz.unsqueeze(dim=0) # <1, vS, 3>
                non_collision, contact = compute_physics(scene_xyz, body_verts, body_faces)
                self.metrics['non_collision'].append(non_collision)
                self.metrics['contact'].append(contact)
            
            ## compute dist.
            if 'dist' in self.eval_metrics:
                obj_mask = sample[i]['info_obj_mask']
                obj_xyz = xyz[obj_mask].unsqueeze(dim=0) # <1, vO, 3>
                
                if obj_xyz.shape[1] != 0:
                    text = sample[i]['c_text']
                    anchor_index = 0 if text.startswith('stand up') else -1
                    anchor_body_verts = body_verts[anchor_index, :, :].unsqueeze(dim=0) # <1, vH, 3>
                    dist = compute_dist_to_obj(obj_xyz, anchor_body_verts, body_faces)
                    self.metrics['dist'].append(dist)

    def eval_apd(self, k_samples: List, dataloader: torch.utils.data.DataLoader, device='cpu'):
        """ Evaluate APD
        """
        for i in range(len(k_samples)):
            x_mask = k_samples[i]['x_mask']
            k_motions = k_samples[i]['k_samples']

            pose_seq = k_motions[:, ~x_mask, :] # (k, l, j * 3)
            pose_seq = dataloader.dataset.denormalize(pose_seq)
            pose_seq = pose_seq[:, :, :self.njoints * 3] # (k, l, j * 3)

            ## compute APD
            apd = compute_pairwise_distance(pose_seq)
            self.metrics['apd'].append(apd)
    
    def eval_humanml(self, sample: List, k_samples: List, dataloader: torch.utils.data.DataLoader, device='cpu'):
        """ Evaluate HumanML metrics
        """
        all_metrics = eval_humanml(sample, k_samples, dataloader, device=device)

        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            for model_name, values in metric_dict.items():
                mean = np.mean(values, axis=0)
                std = np.std(values, axis=0)

                mean_dict[metric_name + '_' + model_name] = mean
        
        for key in mean_dict:
            if isinstance(mean_dict[key], np.ndarray):
                mean_dict[key] = mean_dict[key].tolist()
            if isinstance(mean_dict[key], np.float32) or isinstance(mean_dict[key], np.float64):
                mean_dict[key] = float(mean_dict[key])

            self.metrics['H3D+' + key] = mean_dict[key]
    
    def report(self, save_dir: str):
        """ Report evaluation results

        Args:
            save_dir: save directory
        """
        save_path = os.path.join(save_dir, f'metrics.txt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            for m in self.metrics:
                if m.startswith('_'):
                    continue

                if not m.startswith('H3D'):
                    f.write(f'{m}: {np.mean(self.metrics[m]):.6f}\n')
                else:
                    f.write(f'{m}: {self.metrics[m]}\n')
        with open(save_path.replace('.txt', '.json'), 'w') as f:
            json.dump(self.metrics, f)

def compute_pairwise_distance(x):
    k, n, d = x.shape
    dist = 0
    for j in range(n):
        dist += pairwise_distances(x[:, j, :], x[:, j, :], metric='l2').sum() / (k * (k - 1))
    return dist / n

def compute_physics(points, body_verts, body_faces, contact_threshold=0.05):
    if not torch.is_tensor(points):
        points = torch.tensor(points).float().cuda()
    if not torch.is_tensor(body_verts):
        body_verts = torch.tensor(body_verts).float().cuda()
    if not torch.is_tensor(body_faces):
        body_faces = torch.tensor(body_faces).long().cuda()

    non_collisions = []
    contacts = []
    for f in range(len(body_verts)):
        scene_to_human_sdf, _ = smplx_signed_distance(points, body_verts[f:f+1], body_faces)
        sdf = scene_to_human_sdf.cpu().numpy() # <1, O>

        non_collision = np.sum(sdf <= 0) / sdf.shape[-1]
        if np.sum(sdf > -contact_threshold) > 0:
            contact = 1.0
        else:
            contact = 0.0
        
        non_collisions.append(non_collision)
        contacts.append(contact)
    return sum(non_collisions) / len(non_collisions), sum(contacts) / len(contacts)

def compute_dist_to_obj(points, body_verts, body_faces):
    assert points.shape[0] == body_verts.shape[0] == 1, 'only support batch size = 1'
    if not torch.is_tensor(points):
        points = torch.tensor(points).float().cuda()
    if not torch.is_tensor(body_verts):
        body_verts = torch.tensor(body_verts).float().cuda()
    if not torch.is_tensor(body_faces):
        body_faces = torch.tensor(body_faces).long().cuda()

    object_to_human_sdf, _ = smplx_signed_distance(points, body_verts, body_faces) # <1, O, 3>
    dist_to_obj = min(object_to_human_sdf.max().item(), 0) # inner is positive, outer is negative
    return dist_to_obj

def smplx_signed_distance(object_points, smplx_vertices, smplx_face):
    """ Compute signed distance between query points and mesh vertices.
    
    Args:
        object_points: (B, O, 3) query points in the mesh.
        smplx_vertices: (B, H, 3) mesh vertices.
        smplx_face: (F, 3) mesh faces.
    
    Return:
        signed_distance_to_human: (B, O) signed distance to human vertex on each object vertex
        closest_human_points: (B, O, 3) closest human vertex to each object vertex
        signed_distance_to_obj: (B, H) signed distance to object vertex on each human vertex
        closest_obj_points: (B, H, 3) closest object vertex to each human vertex
    """
    # compute vertex normals
    smplx_face_vertices = smplx_vertices[:, smplx_face]
    e1 = smplx_face_vertices[:, :, 1] - smplx_face_vertices[:, :, 0]
    e2 = smplx_face_vertices[:, :, 2] - smplx_face_vertices[:, :, 0]
    e1 = e1 / torch.norm(e1, dim=-1, p=2).unsqueeze(-1)
    e2 = e2 / torch.norm(e2, dim=-1, p=2).unsqueeze(-1)
    smplx_face_normal = torch.cross(e1, e2)     # (B, F, 3)

    # compute vertex normal
    smplx_vertex_normals = torch.zeros(smplx_vertices.shape).float().cuda()
    smplx_vertex_normals.index_add_(1, smplx_face[:,0], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,1], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,2], smplx_face_normal)
    smplx_vertex_normals = smplx_vertex_normals / torch.norm(smplx_vertex_normals, dim=-1, p=2).unsqueeze(-1)

    # compute paired distance of each query point to each face of the mesh
    pairwise_distance = torch.norm(object_points.unsqueeze(2) - smplx_vertices.unsqueeze(1), dim=-1, p=2)    # (B, O, H)
    
    # find the closest face for each query point
    distance_to_human, closest_human_points_idx = pairwise_distance.min(dim=2)  # (B, O)
    closest_human_point = smplx_vertices.gather(1, closest_human_points_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, O, 3)
    query_to_surface = closest_human_point - object_points
    query_to_surface = query_to_surface / torch.norm(query_to_surface, dim=-1, p=2).unsqueeze(-1)
    closest_vertex_normals = smplx_vertex_normals.gather(1, closest_human_points_idx.unsqueeze(-1).repeat(1, 1, 3))
    same_direction = torch.sum(query_to_surface * closest_vertex_normals, dim=-1)
    signed_distance_to_human = same_direction.sign() * distance_to_human    # (B, O)
    
    # find signed distance to object from human
    # signed_distance_to_object = torch.zeros([pairwise_distance.shape[0], pairwise_distance.shape[2]]).float().cuda()-10  # (B, H)
    # signed_distance_to_object, closest_obj_points_idx = torch_scatter.scatter_max(signed_distance_to_human, closest_human_points_idx, out=signed_distance_to_object)
    # closest_obj_points_idx[closest_obj_points_idx == pairwise_distance.shape[1]] = 0
    # closest_object_point = object_points.gather(1, closest_obj_points_idx.unsqueeze(-1).repeat(1,1,3))
    # return signed_distance_to_human, closest_human_point, signed_distance_to_object, closest_object_point, smplx_vertex_normals
    return signed_distance_to_human, closest_human_point

def create_evaluator(cfg: DictConfig, *args, **kwargs):
    """ Create evaluator

    Args:
        cfg: configuration dict

    Returns:
        Evaluator
    """
    return Evaluator.get(cfg.evaluator.name)(cfg, *args, **kwargs)

def visualize_joints(joints, mask, scene_path, scene_trans, pc_xyz):
    from utils.visualize import skeleton_to_mesh
    from smplkit.constants import SKELETON_CHAIN
    kinematic_chain = SKELETON_CHAIN.SMPLH['kinematic_chain'] # remove the hands, jaw, and eyes

    skeleton = joints.reshape(-1, 22, 3)
    skeleton = skeleton[~mask]
    meshes = skeleton_to_mesh(skeleton, kinematic_chain, 22)

    appendix_meshes = [trimesh.creation.axis(origin_size=0.05)]
    if scene_path is not None and scene_path != '':
        scene_mesh = trimesh.load(scene_path, process=False)
        if len(scene_trans.shape) == 1:
            scene_mesh.apply_translation(scene_trans)
        else:
            scene_mesh.apply_transform(scene_trans)
        appendix_meshes.append(scene_mesh)
    
    trimesh.Scene(meshes + appendix_meshes).show()


    pcd = trimesh.PointCloud(vertices=pc_xyz)
    trimesh.Scene(meshes + [pcd]).show()

def visualize_smplx(body_verts, body_faces, scene_path, scene_trans, pc_xyz):
    meshes = []
    for i in range(0, len(body_verts), 10):
        verts = body_verts[i].cpu().numpy()
        faces = body_faces.cpu().numpy()

        meshes.append(trimesh.Trimesh(vertices=verts, faces=faces))   

    appendix_meshes = [trimesh.creation.axis(origin_size=0.05)]
    if scene_path is not None and scene_path != '':
        scene_mesh = trimesh.load(scene_path, process=False)
        if len(scene_trans.shape) == 1:
            scene_mesh.apply_translation(scene_trans)
        else:
            scene_mesh.apply_transform(scene_trans)
        appendix_meshes.append(scene_mesh)
    trimesh.Scene(meshes + appendix_meshes).show()

    pcd = trimesh.PointCloud(vertices=pc_xyz)
    trimesh.Scene(meshes + [pcd]).show()
