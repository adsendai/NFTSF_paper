# Pre-trained Checkpoints

Pre-trained model weights are **not** committed to this repository due to file
size. This file describes how to obtain them and where to place them.

---

## Download

```
[https://anonymous.4open.science/r/REPLACE_ME/](https://zenodo.org/records/20030944?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk3MjkzYmE3LTA3NTUtNDQxZC1hZmViLWU3NWJjMmQ1YWI3ZiIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZmUyNzE0YjZiMmZlMDU5NTM2MTNjMGVmODU5ODliYSJ9.ZXU5lQMF24yaN4w1vDW01_faJ4oWQb4fCDMiBbeZVTqJN87LQz7Hv5CM-I7S22YmEge1kfFpVDDDDh3reOOKTw)
```

---

## Expected directory layout after download

```
checkpoints/
├── ar_K6_H64_tb30_lr1e-3_ep1000_np100_nf100/
│   ├── linear_gaussian/
│   │   ├── model_<timestamp>.pth
│   │   ├── norm_stats_<timestamp>.npz
│   │   └── config_<timestamp>.json
│   ├── single_well/
│   ├── double_well/
│   ├── alanine_phi/
│   ├── sw_sle_em/
│   ├── sw_gle_oe_em/
│   ├── dw_sle_em/
│   └── dw_gle_oe_em/
└── ar_encoder_full_K6_H64_tb30_lr1e-3_ep1000_np100_nf100/
    └── ...
```

The directory name encodes the run tag produced automatically by
`run_full_pipeline.sh` from the hyperparameter configuration.

---

## Using a checkpoint

Point `test_model.py` at the downloaded `.pth` file:

```bash
python test_model.py \
    --model_path checkpoints/ar_K6_H64_tb30_lr1e-3_ep1000_np100_nf100/single_well/model_<timestamp>.pth \
    --norm_stats_path checkpoints/ar_K6_H64_tb30_lr1e-3_ep1000_np100_nf100/single_well/norm_stats_<timestamp>.npz \
    --data_path data/single_well_test.npy \
    --data_format multi_sim \
    --n_past 100 --n_future 100 \
    --n_samples 500 \
    --output_dir results/single_well_eval
```

Pass `--config_path checkpoints/.../config_<timestamp>.json` to automatically
reconstruct the correct model architecture without specifying `--model_variant`
manually.
