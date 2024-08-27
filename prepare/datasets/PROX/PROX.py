from prepare.datasets.dataset import BaseDataset

import os
import glob
import json
import pickle
import torch
import trimesh
import numpy as np
import smplkit as sk
from tqdm import tqdm
from natsort import natsorted

class PROX(BaseDataset):
    def __init__(self, data_dir: str) -> None:
        super(PROX, self).__init__(data_dir)

        self.save_dir = './data/PROX/motions'
        self._female_subjects_ids = [162, 3452, 159, 3403]
        self.cam_trans = {}

        cam_trans_jsons = glob.glob(os.path.join('./data/PROX/cam2world', '*.json'))
        cam_trans_jsons = [f for f in cam_trans_jsons if '_' not in f]

        for cam_json in cam_trans_jsons:
            with open(cam_json, 'r') as f:
                trans = np.array(json.load(f))
            s = cam_json.split('/')[-1].split('.')[0]
            self.cam_trans[s] = trans.astype(np.float32)

        if not os.path.exists('./data/PROX/normalize_to_center.json'):
            scenes = self.cam_trans.keys()
            normalize_to_center = {}
            for s in scenes:
                ply = trimesh.load(os.path.join('./data/PROX/scenes', f'{s}.ply'), process=False)
                scene_verts = ply.vertices.astype(np.float32)
                x = scene_verts[:, 0].mean()
                y = scene_verts[:, 1].mean()
                z = np.percentile(scene_verts[:, 2], 2)
                m = np.eye(4)
                m[0:3, -1] = [-x, -y, -z]
                normalize_to_center[s] = m.tolist()

            with open('./data/PROX/normalize_to_center.json', 'w') as f:
                json.dump(normalize_to_center, f)
        else:
            normalize_to_center = json.load(open('./data/PROX/normalize_to_center.json', 'r'))
        self.normalize_to_center = {s: np.array(normalize_to_center[s], dtype=np.float32) for s in normalize_to_center}

    def process(self) -> None:
        print(f'Processing PROX dataset in {self.data_dir} ...')

        os.makedirs(self.save_dir, exist_ok=True)

        sequences = os.listdir(self.data_dir)
        sequences = [s for s in sequences if os.path.isdir(os.path.join(self.data_dir, s))]
        for sequence in tqdm(sequences):
            pkls = natsorted(glob.glob(os.path.join(self.data_dir, sequence, 'results', '*', '000.pkl')))
            
            scene_id, subject_id, _ = sequence.split('_')
            subject_id = int(subject_id)
            if subject_id in self._female_subjects_ids:
                subject_gender = 'female'
            else:
                subject_gender = 'male'

            pose_params = []
            betas_params = []
            for pkl in pkls:
                if not os.path.exists(pkl):
                    continue

                with open(pkl, 'rb') as fp:
                    ## keys: ['pose_embedding', 'camera_rotation', 'camera_translation', 'betas', 
                    ## 'global_orient', 'transl', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 
                    ## 'leye_pose', 'reye_pose', 'expression', 'body_pose']
                    param = pickle.load(fp)
                
                # print(param['transl'].shape) # <1, 3>
                # print(param['global_orient'].shape) # <1, 3>
                # print(param['betas'].shape) # <1, 10>
                # print(param['body_pose'].shape) # <1, 63>
                # print(param['left_hand_pose'].shape) # <1, 12>
                # print(param['right_hand_pose'].shape) # <1, 12>
                
                ## We fix the scene and transform the smplx body with the camera transformation matrix,
                ## which is different from the PROX official code tranforming scenes.
                ## So we first need to compute the body pelvis location, see more demonstration at 
                ## https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0
                if subject_gender == 'female':
                    body_model = self.body_model_female
                elif subject_gender == 'male':
                    body_model = self.body_model_male
                output = body_model(
                    transl = torch.tensor(param['transl']),
                    orient = torch.tensor(param['global_orient']),
                    betas = torch.tensor(param['betas']),
                    body_pose = torch.tensor(param['body_pose']),
                    left_hand_pose = torch.tensor(param['left_hand_pose']),
                    right_hand_pose = torch.tensor(param['right_hand_pose']),
                    expression = torch.tensor(param['expression']),
                    jaw_pose = torch.tensor(param['jaw_pose']),
                    leye_pose = torch.tensor(param['leye_pose']),
                    reye_pose = torch.tensor(param['reye_pose']),
                )

                cur_transl, cur_orient = sk.utils.matrix_to_parameter(
                    T = torch.from_numpy(self.normalize_to_center[scene_id] @ self.cam_trans[scene_id]),
                    trans = torch.from_numpy(param['transl']),
                    orient = torch.from_numpy(param['global_orient']),
                    pelvis=output.joints[:, 0, :]
                )
                cur_transl = cur_transl.numpy()
                cur_orient = cur_orient.numpy()
                
                left_hand_pose = output.left_hand_pose.numpy()
                right_hand_pose = output.right_hand_pose.numpy()

                pose_param = np.concatenate([
                    cur_transl, cur_orient, param['body_pose'], left_hand_pose, right_hand_pose
                ], axis=1)

                pose_params.append(pose_param)
                betas_params.append(param['betas'])
            
            pose_params = np.concatenate(pose_params, axis=0)
            betas_params = np.concatenate(betas_params, axis=0)
            betas_params = betas_params.mean(axis=0)

            with open(os.path.join(self.save_dir, f'{sequence}.pkl'), 'wb') as fp:
                pickle.dump((pose_params, betas_params), fp)   
