
# Structure

The detailed structure is as follows:

```bash
- afford-motion/
  - body_models/
    - smplx/
      - SMPLX_NEUTRAL.npz
      - ...
  - data/
    - custom/
    - eval/
    - H3D/
    - HUMANISE/
    - HumanML3D/
    - PROX/
    - POINTTRANS_C_N8192_E300/
    - Mean_Std_*
    - ...
  - outputs/
    - CDM-Perceiver-H3D/                # pre-trained ADM model on original HumanML3D dataset
    - CDM-Perceiver-HUMANISE-step200k/  # pre-trained ADM model on HUMANISE dataset
    - CDM-Perceiver-ALL/                # pre-trained ADM model on all datasets for novel set evaluation
    - CMDM-Enc-H3D-mixtrain0.5/         # pre-trained CMDM model on original HumanML3D dataset
    - CMDM-Enc-HUMANISE-step400k        # pre-trained CMDM model on HUMANISE dataset
    - CMDM-Enc-ALL/                     # pre-trained CMDM model on all datasets for novel set evaluation
  - configs/
  - datasets/
  - ...
```

# Process Data

The following process is to prepare the data for training model using HumanML3D, HUMANISE, and PROX together, corresponding to the evaluation on Novel Evaluation Set.

We also use the processed HUMANISE data to train and evaluate on HUMANISE benchmark.

## 1. Process motion of each dataset

### HumanML3D

After download the AMASS dataset (both SMPL+H and SMPL-X versions), run the following commands:

```bash
python prepare/process.py --dataset HumanML3D --data_dir ${YOUR_PATH}/amass/smplx_neutral
```
**The SMPL+H data should be put into `${YOUR_PATH}/amass/smplh`. The SMPL-X data should be put into `${YOUR_PATH}/amass/smplx_neutral`.**

Copy the original `texts.zip` into `./data/HumanML3D` folder and unzip it.

### Process HUMANISE

After download the HUMANISE dataset, run the following commands:

```bash
python prepare/process.py --dataset HUMANISE --data_dir ${YOUR_PATH}/HUMANISE
```

**The `align_data_release` folder and `pure_motion` folder should both be put into `${YOUR_PATH}/HUMANISE`.**

### Process PROX

We use the refined version of PROX's per-frame SMPL-X parameters from LEMO. Please download the the PROX scene and cam2world data and LEMO motion data.

Move the PROX scene and cam2world data into `./data/PROX` folder.

Then run the following commands to process LEMO (motion):

```bash
python prepare/process.py --dataset PROX --data_dir ${YOUR_PATH}/LEMO/PROX_temporal/PROX_temporal/PROXD_temp
```

## 2. Convert SMPL-X to vectorized representations (joint positions)

```bash
python prepare/smplx_to_vec.py --dataset ${DATASET}
# e.g., python prepare/smplx_to_vec.py --dataset PROX
```
## 3. Process the scene point cloud

```bash
python prepare/process_scene.py
```

## 4. Generate Contact Data

```bash
python prepare/generate_contact_data.py --random_segment
```

## 5. Re-split the dataset

```bash
python prepare/split.py
```

## Others

- Generate target mask for HUMANISE dataset (used in evaluation):
  
```bash
python prepare/generate_target_object_mask.py
```