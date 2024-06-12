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
            diffusion.steps=500 \
            task=contact_motion_gen \
            model=cmdm \
            model.arch='trans_enc' \
            model.time_emb_dim=128 \
            task.dataset.sigma=0.8 \
            task.dataset.sets=["HUMANISE"] \
            task.evaluator.k_samples=0 \
            task.evaluator.eval_nbatch=32 \
            task.evaluator.num_k_samples=320 \
            task.test.contact_folder=${CONT}
            
