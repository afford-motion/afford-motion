import os, sys
sys.path.append(os.path.abspath('.'))
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from utils.eval.word_vectorizer import WordVectorizer

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class HumanML3D(Dataset):
    def __init__(self, split='train', **kwargs) -> None:
        super().__init__()

        self.max_text_len = 20
        self.max_motion_len = 196
        self.min_motion_len = 24

        mean_std = np.load('./data/eval/meta/mean_std.npz')
        self.mean = mean_std['mean']
        self.std = mean_std['std']

        self.split = split
        self.split_file = f'./data/eval/meta/{split}.txt'

        self.w_vectorizer = WordVectorizer(os.path.join('./data/eval/glove'), 'our_vab')
        
        ## load data
        id_list = []
        with open(self.split_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            id_list.append(int(line))
        
        self.all_data = []

        anno = pd.read_csv('./data/HumanML3D/contact_motion/anno.csv')
        for i in tqdm(range(len(anno))):
            if i not in id_list:
                continue

            desc = anno.loc[i]['utterance']
            desc = [] if type(desc) != str or desc == '' else desc.split('$$')
            token = anno.loc[i]['others']
            token = [] if type(token) != str or token == '' else token.split('$$')
            token = [token.split(' ') for token in token]
            assert len(desc) == len(token)
            text_list = list(zip(desc, token))

            motion = np.load(f'./data/HumanML3D/contact_motion/motions/{i:0>5}.npy')
            motion = motion.reshape(motion.shape[0], -1)
            if motion.shape[0] < self.min_motion_len or motion.shape[0] > self.max_motion_len:
                continue
            self.all_data.append((i, motion, len(motion), text_list))
        
        self.indices = list(range(len(self.all_data)))
        print(len(self.indices))
        if self.split == 'train' or self.split == 'all':
            random.shuffle(self.indices)
        if self.split == 'test':
            random.seed(0)
            random.shuffle(self.indices)
    
    def normalize(self, pose_seq: np.ndarray) -> np.ndarray:
        """ Normalize pose sequence
        
        Args:
            pose_seq: a numpy array of pose sequence
        
        Return:
            Normalized pose sequence
        """
        return (pose_seq - self.mean) / self.std
    
    def denormalize(self, pose_seq: np.ndarray) -> np.ndarray:
        """ Denormalize pose sequence (usually for visualization)

        Args:
            pose_seq: a numpy array of normalized pose sequence
        
        Return:
            Denormalized pose sequence
        """
        return pose_seq * self.std + self.mean
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        if self.indices is None:
            index = idx
        else:
            index = self.indices[idx]
        
        i, motion, m_len, text_list = self.all_data[index]

        ## text
        desc, tokens = random.choice(text_list)
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        ## motion
        motion = self.normalize(motion)
        if m_len < self.max_motion_len:
            motion = np.concatenate([
                motion, np.zeros((self.max_motion_len - m_len, motion.shape[1]))], axis=0)
        
        return word_embeddings, pos_one_hots, desc, sent_len, motion, m_len, '_'.join(tokens)


class GenHumanML3D(Dataset):

    def __init__(self, sample, dataloader, **kwargs):
        self.dataset = sample
        self.dataloader = dataloader
        self.w_vectorizer = WordVectorizer(os.path.join('./data/eval/glove'), 'our_vab')

        self.max_text_len = 20

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]

        motion, motion_mask, desc, token = data['sample'], data['x_mask'], data['c_text'], data['info_tokens']

        m_len = (~motion_mask).sum()
        tokens = token.split(' ')
        
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, desc, sent_len, motion, m_len, '_'.join(tokens)

class GenMMHumanML3D(Dataset):
    def __init__(self, k_samples, dataloader, **kwargs) -> None:
        self.dataset = k_samples
        self.dataloader = dataloader
        self.w_vectorizer = WordVectorizer(os.path.join('./data/eval/glove'), 'our_vab')
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]

        k_motions = data['k_samples']
        motion_mask = data['x_mask']

        m_lens = []
        for i in range(len(k_motions)):
            m_lens.append((~motion_mask).sum())
        m_lens = np.array(m_lens, dtype=np.int32)

        return k_motions, m_lens

def get_dataset_loader(batch_size=32, split='train', **kwargs):

    dataset = HumanML3D(split=split, **kwargs)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    return loader

def get_gen_dataset_loader(samples, k_samples, dataloader, batch_size=32):
    dataset = GenHumanML3D(samples, dataloader)
    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)

    mm_dataset = GenMMHumanML3D(k_samples, dataloader)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1)
    
    return  motion_loader, mm_motion_loader

if __name__ == '__main__':
    dataloder = get_dataset_loader(batch_size=32, split='test')
    print(len(dataloder))
    for data in dataloder:
        word_embeddings, pos_one_hots, desc, sent_len, motion, m_len, tokens = data
        print(word_embeddings.shape)
        print(pos_one_hots.shape)
        print(desc)
        print(sent_len)
        print(motion.shape)
        print(m_len)
        print(tokens)
        exit(0)
