EXP_NAME=$1
PORT=$2

if [ -z "$PORT" ]
then
    PORT=29500
fi

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} train_ddp.py \
            hydra/job_logging=none hydra/hydra_logging=none \
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
            