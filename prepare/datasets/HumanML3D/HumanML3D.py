
from prepare.datasets.dataset import BaseDataset

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


class HumanML3D(BaseDataset):
    def __init__(self, data_dir: str) -> None:
        super(HumanML3D, self).__init__(data_dir)

        index_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'index.csv',
        )

        self.smplh_dir = self.data_dir.replace('smplx_neutral', 'smplh')
        self.index_file = pd.read_csv(index_path)
        self.total_amount = self.index_file.shape[0]
        self.fps = 20
        self.num_betas = 10
    
    def process(self) -> None:
        print(f'Processing HumanML3D dataset in {self.data_dir} ...')

        save_dir = './data/HumanML3D/motions'
        os.makedirs(save_dir, exist_ok=True)
        for i in tqdm(range(self.total_amount)):
            source_path = self.index_file.loc[i]['source_path']
            new_name = self.index_file.loc[i]['new_name']
            start_frame = self.index_file.loc[i]['start_frame']
            end_frame   = self.index_file.loc[i]['end_frame']

            src_smplh_path = os.path.join(
                self.smplh_dir,
                '/'.join(source_path.split('/')[2:])
            ).replace('.npy', '.npz')

            src_smplx_path = os.path.join(
                self.data_dir,
                '/'.join(source_path.split('/')[2:])
            ).replace('poses.npy', 'stageii.npz').replace(' ', '_')


            if 'humanact12' in src_smplx_path: # no smplx data of humanact12
                continue
            
            if not os.path.exists(src_smplx_path):
                print(f"Not exist, {src_smplx_path} - {new_name}")
                continue

            bdata = np.load(src_smplx_path, allow_pickle=True)
            trans = bdata['trans']
            root_orient = bdata['root_orient']
            betas = bdata['betas'][:self.num_betas]
            poses = bdata['poses']
            pose_body = bdata['pose_body']
            pose_hand = bdata['pose_hand']

            try:
                fps = np.load(src_smplh_path, allow_pickle=True)['mocap_framerate'] # some mocap_framerate is wrong in smplx dataset
                frame_number = bdata['trans'].shape[0]
            except:
                print('Error:', src_smplh_path)
                continue

            down_sample = int(fps / self.fps)
            param_seq = []
            for fid in range(0, frame_number, down_sample):
                param = np.concatenate((
                    trans[fid:fid+1],
                    root_orient[fid:fid+1],
                    pose_body[fid:fid+1],
                    pose_hand[fid:fid+1],
                ), axis=-1) # <3 + 3 + 63 + 90>
                param_seq.append(param)
            data = np.concatenate(param_seq, axis=0) # <N, 159>

            if 'humanact12' not in source_path:
                if 'Eyes_Japan_Dataset' in source_path:
                    data = data[3*self.fps:]
                if 'MPI_HDM05' in source_path:
                    data = data[3*self.fps:]
                if 'TotalCapture' in source_path:
                    data = data[1*self.fps:]
                if 'MPI_Limits' in source_path:
                    data = data[1*self.fps:]
                if 'Transitions_mocap' in source_path:
                    data = data[int(0.5*self.fps):]
                data = data[start_frame:end_frame]

            with open(os.path.join(save_dir, new_name.replace('.npy', '.pkl')), 'wb') as fp:
                pickle.dump((data, betas), fp)
            
