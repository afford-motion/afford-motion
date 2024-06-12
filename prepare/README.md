
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