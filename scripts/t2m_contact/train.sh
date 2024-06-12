EXP_NAME=$1

python train.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=500 \
            task=text_to_motion_contact_gen \
            task.dataset.sigma=0.8 \
            task.train.batch_size=64 \
            task.train.max_steps=300000 \
            task.train.save_every_step=100000 \
            model=cdm \
            model.arch=Perceiver \
            model.scene_model.use_scene_model=False \
            model.text_model.max_length=20
