EXP_DIR=$1
EVAL_MODE=$2
SEED=$3

if [ -z "$SEED" ]
then
    SEED=2023
fi

## set evaluation mode, without mm or with mm
if [ "$EVAL_MODE" = "wo_mm" ]
then
    K_SAMPLES=0
    N_BATCH=32
elif [ "$EVAL_MODE" = "w_mm" ]
then
    K_SAMPLES=30
    N_BATCH=4
else
    echo "EVAL_MODE should be wo_mm or w_mm."
    exit 1
fi

python test.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${EXP_DIR} \
            seed=${SEED} \
            output_dir=outputs \
            diffusion.steps=500 \
            task=text_to_motion_contact_gen \
            model=cdm \
            model.arch=Perceiver \
            model.scene_model.use_scene_model=False \
            model.text_model.max_length=20 \
            task.dataset.sigma=0.8 \
            task.evaluator.k_samples=${K_SAMPLES} \
            task.evaluator.eval_nbatch=${N_BATCH} \
            task.evaluator.num_k_samples=128
