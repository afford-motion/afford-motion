import os
import glob
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from natsort import natsorted
import random

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class H3DEvalDataset(Dataset):

    def __init__(self, gt_loader, eval_data_folder, mm_eval_data_folder):

        generated_motion = []
        mm_generated_motions = []

        print('###### Load results from', eval_data_folder)
        print('###### Load mm results from', mm_eval_data_folder)

        ## 
        files = natsorted(glob.glob(os.path.join(eval_data_folder, '*.pkl')))
        random.seed(0)
        random.shuffle(files)
        for f in files:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)

            name = data['name']
            text = data['text']
            tokens = data['tokens']
            motion = data['motion']
            length = data['m_len']

            ## process tokens
            if len(tokens) < gt_loader.dataset.opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (gt_loader.dataset.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:gt_loader.dataset.opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)

            generated_motion.append({
                'motion': motion,
                'length': length,
                'caption': text,
                'tokens': tokens,
                'cap_len': sent_len
                # 'cap_len': len(tokens)
            })
        
        ## 
        files = natsorted(glob.glob(os.path.join(mm_eval_data_folder, '*.pkl')))
        random.seed(0)
        random.shuffle(files)
        for f in files:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)

                name = data['name']
                text = data['text']
                tokens = data['tokens']
                motion = data['motion']
                length = data['m_len']

                mm_motions = [{
                    'motion': motion[i],
                    'length': length,
                } for i in range(len(motion))]

                ## process tokens
                if len(tokens) < gt_loader.dataset.opt.max_text_len:
                    # pad with "unk"
                    tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                    sent_len = len(tokens)
                    tokens = tokens + ['unk/OTHER'] * (gt_loader.dataset.opt.max_text_len + 2 - sent_len)
                else:
                    # crop
                    tokens = tokens[:gt_loader.dataset.opt.max_text_len]
                    tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                    sent_len = len(tokens)

                mm_generated_motions.append({
                    'caption': text,
                    'tokens': tokens,
                    'mm_motions': mm_motions,
                    'cap_len': sent_len
                    # 'cap_len': len(tokens)
                })

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = gt_loader.dataset.w_vectorizer

        self.mean_for_eval = gt_loader.dataset.mean
        self.std_for_eval = gt_loader.dataset.std


    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        ## renorm
        renormed_motion = (motion - self.mean_for_eval) / self.std_for_eval  # according to T2M norms
        motion = renormed_motion

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class MMH3DEvalDataset(Dataset):
    def __init__(self,  motion_dataset):
        self.dataset = motion_dataset.mm_generated_motion

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            # We don't need the following logic because our sample func generates the full tensor anyway:
            # if len(motion) < self.opt.max_motion_length:
            #     motion = np.concatenate([motion,
            #                              np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
            #                              ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens

def get_h3d_eval_dataloader(gt_loader, batch_size, eval_data_folder, mm_eval_data_folder):

    dataset = H3DEvalDataset(gt_loader, eval_data_folder, mm_eval_data_folder)
    mm_dataset = MMH3DEvalDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4, shuffle=False)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    return motion_loader, mm_motion_loader