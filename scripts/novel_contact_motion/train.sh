EXP_NAME=$1

python train.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=1000 \
            task=contact_motion_gen \
            task.train.batch_size=32 \
            task.train.max_steps=600000 \
            task.train.save_every_step=100000 \
            task.train.phase=train \
            task.dataset.sigma=0.8 \
            task.dataset.mix_train_ratio=0.0 \
            task.dataset.train_transforms=['RandomRotation','ApplyTransformCMDM','RandomEraseLang','RandomEraseContact','NumpyToTensor'] \
            model=cmdm \
            model.arch='trans_enc'
            