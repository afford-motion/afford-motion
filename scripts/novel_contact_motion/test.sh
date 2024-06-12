EXP_DIR=$1
CONT=$2
SEED=$3

if [ -z "$SEED" ]
then
    SEED=2023
fi

python test.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${EXP_DIR} \
            seed=${SEED} \
            output_dir=outputs \
            diffusion.steps=1000 \
            task=contact_motion_gen \
            task.dataset.sigma=0.8 \
            task.dataset.name=ContactMotionCustomDataset \
            model=cmdm \
            model.arch='trans_enc' \
            task.evaluator.eval_metrics=['Rprecison','apd','non_collision','contact'] \
            task.evaluator.k_samples=30 \
            task.evaluator.num_k_samples=32 \
            task.evaluator.eval_nbatch=5 \
            task.test.batch_size=16 \
            task.test.contact_folder=${CONT}
