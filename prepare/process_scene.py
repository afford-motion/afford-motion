import os
import trimesh
import glob
import numpy as np
from tqdm import tqdm
from natsort import natsorted

def process_scene(scene_path, out_filename):
    mesh = trimesh.load(scene_path, process=False)
    verts = mesh.vertices
    color = mesh.visual.vertex_colors[:, 0:3] / 127.5 - 1.0

    dataset = scene_path.split('/')[2]
    scene = scene_path.split('/')[-1]
    feat = np.load(os.path.join(f'./data/{dataset}/feat', scene.replace('.ply', '_openscene_feat_distill.npy')))
    assert verts.shape[0] == feat.shape[0]

    np.save(out_filename, np.concatenate([verts, color], axis=1).astype(np.float32))
    
if __name__=='__main__':
    ## process HUMANISE
    os.makedirs('./data/HUMANISE/points/', exist_ok=True)
    paths = natsorted(glob.glob('./data/HUMANISE/scenes/*_00/*_00_vh_clean_2.ply'))
    for scene_path in tqdm(paths):
        try:
            scene_name = scene_path.split('/')[-2]
            out_filename = scene_name+'.npy'
            
            process_scene(scene_path, os.path.join('./data/HUMANISE/points/', out_filename))
        except Exception as e:
            print(e)
            print(scene_name+'ERROR!!')
    print("done!")

    ## process PROX
    os.makedirs('./data/PROX/points/', exist_ok=True)
    paths = natsorted(glob.glob('./data/PROX/scenes/*.ply'))
    for scene_path in tqdm(paths):
        try:
            scene_name = scene_path.split('/')[-1].split('.')[0]
            out_filename = scene_name+'.npy'

            process_scene(scene_path, os.path.join('./data/PROX/points/', out_filename))
        except Exception as e:
            print(e)
            print(scene_name+'ERROR!!')
    print("done!")

    ## process HumanML3D
    os.makedirs('./data/HumanML3D/points/', exist_ok=True)
    paths = natsorted(glob.glob('./data/HumanML3D/scenes/*.ply'))
    for scene_path in tqdm(paths):
        try:
            scene_name = scene_path.split('/')[-1].split('.')[0]
            out_filename = scene_name+'.npy'

            process_scene(scene_path, os.path.join('./data/HumanML3D/points/', out_filename))
        except Exception as e:
            print(e)
            print(scene_name+'ERROR!!')
    print("done!")
    