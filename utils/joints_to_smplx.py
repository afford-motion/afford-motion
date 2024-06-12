import os, sys
sys.path.append(os.path.abspath('./'))
import time
import glob
import pickle
import trimesh
import torch
import torch.nn as nn
import numpy as np
import smplkit as sk
from natsort import natsorted
from tqdm import tqdm

from models.modules import PositionalEncoding
from utils.misc import optimize_params_with_joints
from utils.misc import smplx_neutral_model, get_meshes_from_smplx, get_joints_from_smplx, get_joints_and_meshes_from_smplx

def padding_motion(x, max_length):
    """ Pad one motion sequence to the max_length """
    l, d = x.shape
    if l <= max_length:
        pad_x = np.concatenate([
            x, np.zeros((max_length-l, d), dtype=np.float32)
        ], axis=0)
        mask = np.concatenate([
            np.zeros(l, dtype=bool),
            np.ones(max_length-l, dtype=bool)
        ], axis=0)
    else:
        pad_x = x[:max_length]
        mask = np.zeros(max_length, dtype=bool)
    return pad_x, mask

def padding_motions(x, max_length):
    """ Pad a batch motion sequence to the same length """
    pad_x_all = []
    mask_all = []
    for i in range(len(x)):
        px, mask = padding_motion(x[i], max_length)
        pad_x_all.append(px)
        mask_all.append(mask)
    pad_x_all = np.stack(pad_x_all, axis=0)
    mask_all = np.stack(mask_all, axis=0)
    return pad_x_all, mask_all

class JointsToSMPLX(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.opt_rate = kwargs.get('opt_rate', 5e-2)
        self.opt_steps = kwargs.get('opt_steps', 100)
        print(self.opt_rate, self.opt_steps)

        self.max_length = 196
        self.min_length = 24

        self.njoints = 22
        self.input_feats = self.njoints * 3
        self.output_feats = 3 + self.njoints * 3
        self.time_emb_dim = 256
        self.latent_dim = 256
        self.num_heads = 4
        self.ff_size = 512
        self.dropout = 0.1
        self.activation = 'relu'
        self.num_layers = 2 # 6

        self.positional_encoder = PositionalEncoding(self.time_emb_dim, dropout=self.dropout, max_len=1000)
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_feats, self.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim // 2, self.latent_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(self.latent_dim, self.output_feats)

        transEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation,
                                                        batch_first=True)
        self.TransEncoder = nn.TransformerEncoder(transEncoderLayer,
                                                    num_layers=self.num_layers)

    def load_and_freeze(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print(f'Load model from {model_path}')
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, x_mask=None, *args, **kwargs):
        x = self.input_layer(x)
        x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.TransEncoder(x, src_key_padding_mask=x_mask)
        x = self.output_layer(x)
        return x
    
    def joints_to_params_batch(self, body_model, joints_batch, joints_mask_batch, optimize=False, *args, **kwargs):
        """ Converts joints to smplx params, Support batch size > 1

        If optimize, the program will optimize the smplx parameters with the given joints positions through gradient descent.

        Args:
            body_model: smplx model
            joints_batch: joints sequence <b, l, 22 * 3>, numpy array
            joints_mask_batch: joints mask <b, l>, numpy array
            optimize: whether to optimize the smplx parameters
        
        Returns:
            Return a list of optimized smplx parameters if optimize, each item is a torch.Tensor of shape <l_i, 69>.
            If optimize is False, return a torch.Tensor of shape <b, l, 69>.
        """
        b, l, d = joints_batch.shape
        pred_params = self.forward(joints_batch, joints_mask_batch)

        if optimize:
            opt_params = []
            for i in range(b):
                mask = joints_mask_batch[i]
                gt_joints = joints_batch[i][~mask]

                opt_param = optimize_params_with_joints(
                    body_model, gt_joints, init_params=pred_params[i][~mask],
                    lr=self.opt_rate, steps=self.opt_steps, log=False)
                opt_params.append(opt_param)
        else:
            opt_params = pred_params
        
        return opt_params

class JointsToSMPLXDataset(torch.utils.data.Dataset):

    def __init__(self, phase) -> None:
        self.max_length = 196
        self.min_length = 24
        self.same_betas = True

        self.train_ratio = 0.8

        param_files = natsorted(glob.glob('./data/HumanML3D/motions/0*.pkl'))
        if phase == 'train':
            param_files = param_files[0:int(len(param_files)*self.train_ratio)]
        else:
            param_files = param_files[int(len(param_files)*self.train_ratio):]

        self.params = []
        self.joints = []
        for pf in tqdm(param_files):
            ## smplx parameters
            params = pickle.load(open(pf, 'rb'))

            pose_seq, betas = params    
            transl = pose_seq[:, :3].astype(np.float32)
            orient = pose_seq[:, 3:6].astype(np.float32)
            body_pose = pose_seq[:, 6:69].astype(np.float32)
            self.params.append(np.concatenate([transl, orient, body_pose], axis=1))            

            ## joints positions
            jf = pf.replace('motions', 'motions_pos').replace('.pkl', '.npy')
            joints = np.load(jf).astype(np.float32)
            self.joints.append(joints)

            assert len(transl) == len(joints)
        
        assert len(self.params) == len(self.joints)
    
    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        joint = self.joints[index]
        param = self.params[index]
        # print(joint.shape, param.shape)
        joint, mask = padding_motion(joint, self.max_length)
        # print(joint.dtype, joint.shape, mask.dtype, mask.shape, mask.sum())
        param, mask = padding_motion(param, self.max_length)
        # print(param.dtype, param.shape, mask.dtype, mask.shape, mask.sum())
        
        return joint.astype(np.float32), param.astype(np.float32), mask

def compute_loss(pred_param, param, mask, input_joint,):

    ## param loss
    param_loss = ((pred_param - param) ** 2).sum(dim=-1) # <b, l>
    param_loss = (param_loss * (~mask)).sum(dim=-1) / (~mask).sum(dim=-1) # <b>

    ## auxiliary joints and verts loss
    joints, verts, _ = get_joints_and_meshes_from_smplx(body_model, param)
    joints = joints.detach()
    verts = verts.detach()

    pred_joints, pred_verts, _ = get_joints_and_meshes_from_smplx(body_model, pred_param)

    # b, l = input_joint.shape[:2]
    # input_joint = input_joint.reshape(b, l, -1, 3)
    # j = input_joint.shape[-1]
    # joint_err = ((input_joint - joints) ** 2).sum(dim=-1).sum(dim=-1) / j # <b, l>
    # print(joint_err.shape)
    # joint_err = (joint_err * (~mask)).sum(dim=-1) / (~mask).sum(dim=-1) # <b>
    # print(joint_err.sum())
    # exit(0)

    ## joints loss
    b, l, j, _ = joints.shape
    joint_loss = ((pred_joints - joints) ** 2).sum(dim=-1).sum(dim=-1) / j # <b, l>
    joint_loss = (joint_loss * (~mask)).sum(dim=-1) / (~mask).sum(dim=-1) # <b>

    ## verts loss
    b, l, v, _ = verts.shape
    verts_loss = ((pred_verts - verts) ** 2).sum(dim=-1).sum(dim=-1) / v # <b, l>
    verts_loss = (verts_loss * (~mask)).sum(dim=-1) / (~mask).sum(dim=-1) # <b>

    return param_loss, joint_loss, verts_loss

def train():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    lr = 1e-3
    weight_decay = 1e-2
    max_epochs = 100
    device = 'cuda'
    
    model = JointsToSMPLX().to(device=device)
    train_dataset = JointsToSMPLXDataset('train')
    test_dataset = JointsToSMPLXDataset('test')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print('Train dataset size:', len(train_dataset))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    print('Test dataset size:', len(test_dataset))

    params = []
    nparams = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    print()
    print(f'{len(params)} parameters for optimization.')
    print(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')

    total_step = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for it, data in enumerate(train_dataloader): 
            joint, param, mask = data
            joint = joint.float().to(device)
            param = param.float().to(device)
            mask = mask.bool().to(device)

            ## one step optimization
            optimizer.zero_grad()
            pred_param = model(joint, mask, device=device) # <b, l, d>
            param_loss, joint_loss, vert_loss = compute_loss(pred_param, param, mask, joint)
            loss = param_loss.mean() + joint_loss.mean() * 0.5 + vert_loss.mean() * 0.5
            loss.backward()
            optimizer.step()

            total_step += 1
            if total_step % 100 == 0:
                print(f'Epoch {epoch}, step {total_step:0>5d}, loss: {loss.item():.4f}')
        scheduler.step()

        model.eval()
        total_param_err, total_joint_err, total_vert_err = 0., 0., 0.
        ncase = 0
        with torch.no_grad():
            for it, data in enumerate(test_dataloader):
                joint, param, mask = data
                joint = joint.float().to(device)
                param = param.float().to(device)
                mask = mask.bool().to(device)

                pred_param = model(joint, mask, device=device) # <b, l, d>
                b_param_err, b_joint_err, b_vert_err = compute_loss(pred_param, param, mask, joint)
                total_param_err += b_param_err.sum().item()
                total_joint_err += b_joint_err.sum().item()
                total_vert_err  += b_vert_err.sum().item()
                ncase += len(joint)
        print(f'Epoch {epoch}: test param error: {total_param_err / ncase:.4f}, ' + \
              f'test joint error: {total_joint_err / ncase:.4f}, test vert error: {total_vert_err / ncase:.4f}')

        if epoch % 20 == 0:
            os.makedirs(os.path.dirname(f'./utils/joints_to_smplx/'), exist_ok=True)
            torch.save(model.state_dict(), f'./utils/joints_to_smplx/{epoch:0>3d}.pt')

def test():
    device = 'cuda'
    model = JointsToSMPLX(opt_rate=2e-2, opt_steps=99).to(device=device)
    model.load_and_freeze('./utils/joints_to_smplx/060.pt')
    test_dataset = JointsToSMPLXDataset('test')
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    for data in dataloader:
        joint, param, mask = data
        joint = joint.float().to(device)
        param = param.float().to(device)
        mask = mask.bool().to(device)

        ## test optimization
        # in_joints, in_params = joint.clone().detach(), param.clone().detach()
        # in_joints, in_params = in_joints[0][~mask[0]], in_params[0][~mask[0]]

        # out_param = optimize_params_with_joints(body_model, in_joints, steps=100, log=False)
        # print(in_joints.shape, out_param.shape)

        # in_joints, in_verts, in_faces = get_joints_and_meshes_from_smplx(body_model, in_params.unsqueeze(0))
        # in_joints, in_verts = in_joints.squeeze(0).cpu().numpy(), in_verts.squeeze(0).cpu().numpy()
        # out_joints, out_verts, out_faces = get_joints_and_meshes_from_smplx(body_model, out_param.unsqueeze(0))
        # out_joints, out_verts = out_joints.squeeze(0).cpu().numpy(), out_verts.squeeze(0).cpu().numpy()
        # print(in_joints.shape, in_verts.shape, in_faces.shape)
        # print(out_joints.shape, out_verts.shape, out_faces.shape)

        # for i in range(len(in_joints)):
        #     S = trimesh.Scene()
        #     S.add_geometry(trimesh.creation.axis())
        #     S.add_geometry(trimesh.PointCloud(vertices=in_joints[i], colors=[255, 0, 0]))
        #     S.add_geometry(trimesh.PointCloud(vertices=out_joints[i], colors=[0, 255, 0]))
        #     S.show()

        #     S = trimesh.Scene()
        #     S.add_geometry(trimesh.creation.axis())
        #     S.add_geometry(trimesh.Trimesh(vertices=in_verts[i], faces=in_faces, vertex_colors=[255, 0, 0]))
        #     S.add_geometry(trimesh.Trimesh(vertices=out_verts[i], faces=out_faces, vertex_colors=[0, 255, 0]))
        #     S.show()

        # exit(0)

        # ## projection
        # pred_param = model(joint, mask, device=device)
        # pred_param = pred_param[0][~mask[0]]
        # pred_verts, faces = get_meshes_from_smplx(body_model, pred_param.unsqueeze(0))
        # pred_verts = pred_verts.squeeze(0).cpu().numpy()

        gt_param = param[0][~mask[0]]
        gt_verts, faces = get_meshes_from_smplx(body_model, gt_param.unsqueeze(0))
        gt_verts = gt_verts.squeeze(0).cpu().numpy()

        # # for i in range(0, len(pred_verts), 10):
        # #     scene = trimesh.Scene()
        # #     scene.add_geometry(trimesh.creation.axis())
        # #     scene.add_geometry(trimesh.Trimesh(vertices=pred_verts[i], faces=faces, vertex_colors=[255, 0, 0]))
        # #     scene.add_geometry(trimesh.Trimesh(vertices=gt_verts[i], faces=faces, vertex_colors=[0, 255, 0]))
        # #     scene.show()
        # #     break
        
        # ## optimization
        # gt_joint = joint[0][~mask[0]] # <l, 22 * 3>
        # # print(gt_joint.shape, pred_param.shape)
        # opt_param = optimize_params_with_joints(body_model, gt_joint, lr=5e-2, init_params=pred_param, steps=50, log=True)
        # # print(opt_param.shape)
        # opt_verts, faces = get_meshes_from_smplx(body_model, opt_param.unsqueeze(0))
        # opt_verts = opt_verts.squeeze(0).cpu().numpy()

        # for i in range(0, len(opt_verts), 10):
        #     scene = trimesh.Scene()
        #     scene.add_geometry(trimesh.creation.axis())
        #     scene.add_geometry(trimesh.Trimesh(vertices=opt_verts[i], faces=faces, vertex_colors=[255, 0, 0]))
        #     scene.add_geometry(trimesh.Trimesh(vertices=gt_verts[i], faces=faces, vertex_colors=[0, 255, 0]))
        #     scene.add_geometry(trimesh.Trimesh(vertices=pred_verts[i], faces=faces, vertex_colors=[0, 0, 255]))
        #     scene.show()
        #     break
        # exit()

        ## test joints_to_params_batch
        opt_param = model.joints_to_params_batch(body_model, joint, mask, optimize=True)[0]
        opt_verts, faces = get_meshes_from_smplx(body_model, opt_param.unsqueeze(0))
        opt_verts = opt_verts.squeeze(0).cpu().numpy()

        for i in range(0, len(opt_verts), 10):
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.creation.axis())
            scene.add_geometry(trimesh.Trimesh(vertices=opt_verts[i], faces=faces, vertex_colors=[255, 0, 0]))
            scene.add_geometry(trimesh.Trimesh(vertices=gt_verts[i], faces=faces, vertex_colors=[0, 255, 0]))
            scene.show()
        exit()


if __name__ == '__main__':
    SEED = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    body_model = smplx_neutral_model.cuda()

    # train()
    test()
