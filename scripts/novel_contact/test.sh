EXP_DIR=$1
SEED=$2

if [ -z "$SEED" ]
then
    SEED=2023
fi

python test.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${EXP_DIR} \
            seed=${SEED} \
            output_dir=outputs \
            diffusion.steps=500 \
            task=contact_gen \
            model=cdm \
            model.arch=Perceiver \
            task.dataset.sigma=0.8 \
            task.dataset.name=ContactMapCustomDataset \
            task.evaluator.eval_metrics=[] \
            task.evaluator.k_samples=30 \
            task.evaluator.num_k_samples=32 \
            task.evaluator.eval_nbatch=5 \
            task.test.batch_size=16
