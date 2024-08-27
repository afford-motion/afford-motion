import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from natsort import natsorted
from tqdm import tqdm
import json
import trimesh
from plyfile import PlyData, PlyElement

def load_humanise(min_horizon: int, max_horizon: int, **kwargs: Dict) -> Tuple:
    """ Load humanise dataset

    Args:
        min_horizon: minimum horizon for motion sequence
        max_horizon: maximum horizon for motion sequence
    
    Return:
        motions: list of motion-condition pairs, each contains (pose_seq, [text], (scene_id, scene_trans), other_info)
    """
    motion_paths = natsorted(glob.glob('./data/HUMANISE/motions_pos/*.npy'))
    anno_file = pd.read_csv('./data/HUMANISE/annotations.csv')
    assert anno_file.shape[0] == len(motion_paths)
    
    data = []
    for motion_path in tqdm(motion_paths, desc='Loading HUMANISE..'):
        pose_seq = np.load(motion_path)
        index = int(motion_path.split('/')[-1].split('.')[0])

        if len(pose_seq) < min_horizon or len(pose_seq) > max_horizon:
            continue
        
        texts = [anno_file.loc[index]['text']]
        scene_id = anno_file.loc[index]['scene_id']
        obj_semantic_label = anno_file.loc[index]['object_semantic_label']        
        data.append((texts, scene_id, obj_semantic_label))
    
    return data

def read_ply_xyzrgb(filename):
    """ read XYZRGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

CLASS_NAMES = {
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

def get_raw2scannet_label_map():
    # lines = [line.rstrip() for line in open('scannet-labels.combined.tsv')]
    lines = [line.rstrip() for line in open('../point_transformer.scannet/preprocessing/scannetv2-labels.combined.tsv')]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(CLASS_NAMES.keys())
        elements = lines[i].split('\t')
        # raw_name = elements[0]
        # nyu40_name = elements[6]
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'otherprop'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet
RAW2SCANNET = get_raw2scannet_label_map()

def collect_one_scene_data_label(scene_name):
    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join('./data/HUMANISE/scenes', scene_name)

    # segs
    mesh_seg_filename = os.path.join(data_folder, '%s_vh_clean_2.0.010000.segs.json'%(scene_name))
    with open(mesh_seg_filename) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
        
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)
    
    npoints = 0
    for key in segid_to_pointid:
        npoints += len(segid_to_pointid[key])
    
    # Raw points in XYZRGB
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    points = read_ply_xyzrgb(ply_filename)
    assert len(points) == npoints
    
    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    # annotation_filename = os.path.join(data_folder, '%s.aggregation.json'%(scene_name)) # low-res mesh
    annotation_filename = os.path.join(data_folder, '%s_vh_clean.aggregation.json'%(scene_name)) # high-res mesh
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])

    # Each instance's points
    semantic_labels_arr = np.zeros(len(points)) + 40
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        
        label = RAW2SCANNET[labels[i]]
        label = CLASS_NAMES[label]

        instance_indices = np.array(pointids)
        semantic_labels_arr[instance_indices] = label
    semantic_labels_arr = semantic_labels_arr.astype(np.int64)

    return semantic_labels_arr

PALETTE = np.random.randint(low=0, high=255, size=(41, 3)).astype(np.uint8)

if __name__ == '__main__':
    data = load_humanise(24, 196)

    try:
        import pickle
        with open('./data/HUMANISE/semantics.pkl', 'rb') as f:
            scenes = pickle.load(f)
    except:
        scenes = {}
        for i in tqdm(range(707)):
            scenes[f'scene{i:04d}_00'] = collect_one_scene_data_label(f'scene{i:04d}_00')
        
        with open('./data/HUMANISE/semantics.pkl', 'wb') as f:
            pickle.dump(scenes, f)
    
    ## test scene
    # scene_id = 'scene0000_00'
    # mesh = trimesh.load(f'./data/HUMANISE/scenes/{scene_id}/{scene_id}_vh_clean_2.ply', process=False)
    # points = mesh.vertices
    # colors = np.zeros((len(points), 3)).astype(np.uint8)
    # semantic_labels = scenes[scene_id]
    # for i in range(len(points)):
    #     colors[i, 0:3] = PALETTE[semantic_labels[i]]
    # trimesh.PointCloud(points, colors=colors).show()

    ## process each contact
    for i in tqdm(range(len(data))):
        texts, scene_id, obj_semantic_label = data[i]

        contact = np.load(f'./data/HUMANISE/contact_motion/contacts/{i:0>5d}.npz')
        points = contact['points'][:, 0:3]
        mask = contact['mask']

        sem = scenes[scene_id]
        sem = sem[mask]
        target_mask = (sem == obj_semantic_label)
        if target_mask.sum() == 0:
            print('no target object in scene', i)
            colors = np.zeros((len(points), 3)).astype(np.uint8)
            colors[target_mask, 0:3] = np.array([255, 0, 0])
            trimesh.PointCloud(points, colors=colors).show()

            # colors = np.zeros((len(points), 3)).astype(np.uint8)
            # for i in range(len(colors)):
            #     colors[i, 0:3] = np.array(PALETTE[sem[i]])
            # trimesh.PointCloud(vertices=points, colors=colors).show()

        if i % 1000 == 0:
            colors = np.zeros((len(points), 3)).astype(np.uint8)
            colors[target_mask, 0:3] = np.array([255, 0, 0])
            trimesh.PointCloud(points, colors=colors).show()
        
        ## save mask
        os.makedirs(f'./data/HUMANISE/contact_motion/target_mask', exist_ok=True)
        np.save(f'./data/HUMANISE/contact_motion/target_mask/{i:0>5d}.npy', target_mask)
