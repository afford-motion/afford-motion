## Code modified from Text-to-Motion and MDM
import torch
from collections import OrderedDict
from utils.eval.metrics import *
from utils.eval.get_data import get_dataset_loader, get_gen_dataset_loader
from utils.eval.evaluator_wrapper import EvaluatorWrapper

def evaluate_matching_score(eval_wrapper, motion_loaders):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, diversity_times):
    eval_dict = OrderedDict({})
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity

    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, mm_num_times):
    eval_dict = OrderedDict({})
    
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        
        eval_dict[model_name] = multimodality
    return eval_dict

def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, diversity_times, mm_num_times, run_mm=False):
    all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                'R_precision': OrderedDict({}),
                                'FID': OrderedDict({}),
                                'Diversity': OrderedDict({}),
                                'MultiModality': OrderedDict({})})
    
    motion_loaders = {}
    mm_motion_loaders = {}
    motion_loaders['ground truth'] = gt_loader
    for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
        print('\n' + motion_loader_name + '\n')
        motion_loader, mm_motion_loader = motion_loader_getter()
        motion_loaders[motion_loader_name] = motion_loader
        mm_motion_loaders[motion_loader_name] = mm_motion_loader

    mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders)

    fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict)

    div_score_dict = evaluate_diversity(acti_dict, diversity_times)

    if run_mm:
        mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, mm_num_times)

    for key, item in mat_score_dict.items():
        if key not in all_metrics['Matching Score']:
            all_metrics['Matching Score'][key] = [item]
        else:
            all_metrics['Matching Score'][key] += [item]

    for key, item in R_precision_dict.items():
        if key not in all_metrics['R_precision']:
            all_metrics['R_precision'][key] = [item]
        else:
            all_metrics['R_precision'][key] += [item]

    for key, item in fid_score_dict.items():
        if key not in all_metrics['FID']:
            all_metrics['FID'][key] = [item]
        else:
            all_metrics['FID'][key] += [item]

    for key, item in div_score_dict.items():
        if key not in all_metrics['Diversity']:
            all_metrics['Diversity'][key] = [item]
        else:
            all_metrics['Diversity'][key] += [item]
    if run_mm:
        for key, item in mm_score_dict.items():
            if key not in all_metrics['MultiModality']:
                all_metrics['MultiModality'][key] = [item]
            else:
                all_metrics['MultiModality'][key] += [item]
    
    return all_metrics


def eval_humanml(samples, k_samples, dataloader, device='cpu'):
    batch_size = 16
    dataset = 'humanml'
    split = 'test'

    mm_num_times = 10
    diversity_times = 32
    run_mm = len(k_samples) > 0

    gt_loader = get_dataset_loader(batch_size=batch_size, split=split)
    # gen_loader = get_gen_dataset_loader(samples, k_samples, gt_loader, batch_size=batch_size)

    eval_motion_loaders = {
        'vald': lambda: get_gen_dataset_loader(samples, k_samples, gt_loader, batch_size=batch_size)
    }

    eval_wrapper = EvaluatorWrapper(dataset, device)
    all_metrics = evaluation(eval_wrapper, gt_loader, eval_motion_loaders, diversity_times, mm_num_times, run_mm=run_mm)

    return all_metrics
