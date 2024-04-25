EXP_NAME=$1

python train.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=500 \
            task=contact_gen \
            task.train.batch_size=64 \
            task.train.max_steps=200000 \
            task.train.save_every_step=100000 \
            task.train.phase=train \
            task.dataset.sigma=0.8 \
            task.dataset.sets=["HUMANISE"] \
            model=cdm \
            model.arch=Perceiver
            