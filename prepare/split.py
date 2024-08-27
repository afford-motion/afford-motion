import os
import random
import pandas as pd
from natsort import natsorted

random.seed(0)

def split_HUMANISE():
    train_id = []
    test_id = []
    anno = pd.read_csv(os.path.join('./data/HUMANISE/contact_motion/anno.csv'))
    for i in range(len(anno)):
        scene_id = anno.loc[i]['scene_id']

        if int(scene_id[5:9]) < 600:
            train_id.append(i)
        else:
            test_id.append(i)

    with open('./data/HUMANISE/train.txt', 'w') as f:
        for i in train_id:
            f.write(f'{i:0>6d}\n')
    with open('./data/HUMANISE/test.txt', 'w') as f:
        for i in test_id:
            f.write(f'{i:0>6d}\n')
    with open('./data/HUMANISE/all.txt', 'w') as f:
        for i in range(len(anno)):
            f.write(f'{i:0>6d}\n')

def split_PROX():
    train_id = []
    test_id = []
    anno = pd.read_csv(os.path.join('./data/PROX/contact_motion/anno.csv'))
    for i in range(len(anno)):
        scene_id = anno.loc[i]['scene_id']
        
        if scene_id in ['BasementSittingBooth', 'MPH11', 'MPH112', 'MPH8', 'N0Sofa', 'N3Library', 'N3Office', 'Werkraum']:
            train_id.append(i)
        else:
            test_id.append(i)

    with open('./data/PROX/train.txt', 'w') as f:
        for i in train_id:
            f.write(f'{i:0>6d}\n')
    with open('./data/PROX/test.txt', 'w') as f:
        for i in test_id:
            f.write(f'{i:0>6d}\n')
    with open('./data/PROX/all.txt', 'w') as f:
        for i in range(len(anno)):
            f.write(f'{i:0>6d}\n')

def split_HumanML3D(train_ratio=0.8):
    train_id = []
    test_id = []
    anno = pd.read_csv(os.path.join('./data/HumanML3D/contact_motion/anno.csv'))

    unique_cases_n = len(anno) // 2
    ids = list(range(unique_cases_n))
    with open('./data/HumanML3D/all.txt', 'w') as f:
        for i in ids:
            f.write(f'{i:0>6d}\n')
            f.write(f'{i+unique_cases_n:0>6d}\n')
    
    random.shuffle(ids)
    train_id = ids[:int(len(ids) * train_ratio)]
    test_id = ids[int(len(ids) * train_ratio):]
    train_id = natsorted(train_id)
    test_id = natsorted(test_id)
    with open('./data/HumanML3D/train.txt', 'w') as f:
        for i in train_id:
            f.write(f'{i:0>6d}\n')
            f.write(f'{i+unique_cases_n:0>6d}\n')
    with open('./data/HumanML3D/test.txt', 'w') as f:
        for i in test_id:
            f.write(f'{i:0>6d}\n')
            f.write(f'{i+unique_cases_n:0>6d}\n')

if __name__ == '__main__':
    split_HUMANISE()
    split_PROX()
    split_HumanML3D(train_ratio=0.8)
