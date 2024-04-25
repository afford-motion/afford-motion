import torch
import smplkit as sk

def compute_repr_dimesion(data_repr: str) -> int:
    if data_repr == 'smplx_no_hands':
        return 3 + 3 + 63 # global translation + global orientation + 21 joints rotation
    elif data_repr == 'pos':
        return 22 * 3
    elif data_repr == 'pos_rot':
        return 22 * 3 + 21 * 3 # 22 joints position and 21 joints rotation
    elif data_repr == 'contact_one_joints':
        return 1
    elif data_repr == 'contact_all_joints':
        return 22
    elif data_repr == 'contact_cont_joints':
        return 6
    elif data_repr == 'contact_pelvis':
        return 1
    elif data_repr == 'h3d':
        return 263
    else:
        raise ValueError(f"Unknown data representation: {data_repr}")

smplx_neutral_model = sk.SMPLXLayer(gender='neutral', num_betas=10, num_pca_comps=-1)

def get_meshes_from_smplx(body_model_neutral, param):
    b, l, d = param.shape
    param = param.reshape(b * l, d)

    transl = param[:, :3]
    orient = param[:, 3:6]
    body_pose = param[:, 6:69]

    verts = body_model_neutral(
        transl=transl,
        orient=orient,
        body_pose=body_pose,
        return_verts=True
    )
    return verts.reshape(b, l, -1, 3), body_model_neutral.faces

def get_joints_from_smplx(body_model_neutral, param):
    b, l, d = param.shape
    param = param.reshape(b * l, d)

    transl = param[:, :3]
    orient = param[:, 3:6]
    body_pose = param[:, 6:69]

    joints = body_model_neutral(
        transl=transl,
        orient=orient,
        body_pose=body_pose,
        return_joints=True
    )
    joints = joints[:, :22, :] # <nframes, 22, 3>
    return joints.reshape(b, l, 22, 3)

def get_joints_and_meshes_from_smplx(body_model_neutral, param):
    b, l, d = param.shape
    param = param.reshape(b * l, d)

    transl = param[:, :3]
    orient = param[:, 3:6]
    body_pose = param[:, 6:69]

    output = body_model_neutral(
        transl=transl,
        orient=orient,
        body_pose=body_pose,
    )
    joints = output.joints[:, :22, :] # <nframes, 22, 3>
    joints = joints.reshape(b, l, 22, 3)
    verts = output.vertices # <nframes, 10475, 3>
    verts = verts.reshape(b, l, -1, 3)
    return joints, verts, body_model_neutral.faces

def optimize_params_with_joints(body_model, joints, init_params=None, lr=0.05, steps=500, log=False, **kwargs):
    """ Optimize the body parameters with given joints

    Args:
        body_model: SMPLXLayer
        joints: <nframes, 22, 3> or <nframes, 66>
        init_params: <nframes, 69>
        lr: learning rate
        steps: max optimization steps
        log: whether to print the loss
    """
    l = joints.shape[0]
    joints = joints.reshape(l, 22, 3)

    if init_params is None:
        init_params = torch.zeros(l, 69, device=joints.device)
    else:
        init_params = init_params.reshape(l, 69).clone().detach()
    init_params.requires_grad_(True)
    
    optimizer = torch.optim.Adam([init_params], lr=lr)

    step = 0
    while True:
        joints_ = body_model(
            transl=init_params[:, :3],
            orient=init_params[:, 3:6],
            body_pose=init_params[:, 6:69],
            return_joints=True
        )

        if step >= 0.6 * steps:
            loss = compute_optimization_loss(joints_, joints, opt_params=init_params)
        else:
            loss = compute_optimization_loss(joints_, joints)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if log:
            print(loss)
        
        ## max optimization step
        step += 1
        if step >= steps:
            break
    
    return init_params.detach().reshape(l, -1)

def compute_optimization_loss(opt_joints, joints, opt_params=None):
    l, j, _ = joints.shape

    loss = 0.

    ## joints loss
    joints_loss = torch.mean((opt_joints[:, :j, :] - joints) ** 2)
    loss += joints_loss

    ## params loss, smoothness
    if opt_params is not None:
        velocity = opt_params[1:] - opt_params[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        params_loss = torch.mean(acceleration ** 2)
        loss += params_loss * 0.1

    return loss
