# DRIMed

Official PyTorch implementation of **DRIMed**, a deep learning model for safe and personalized medication recommendation on electronic health records (MIMIC-III).

## Overview

DRIMed predicts a safe and effective medication combination for a patient based on their electronic health records (diagnoses, procedures, demographics, and historical medications). The model integrates:

- **IIM (Intra-visit Interaction Module)** — SAMFormer-based channel attention encoder for fusion of diagnosis and procedure entities within each visit.
- **RCE (Temporal Residual Convolutional Encoder)** — multi-scale 1D-CNN (kernel 3 and 5) for sequential patient representation.
- **MKG (Medication Knowledge Graph memory)** — GCN-based drug memory bank built from the EHR co-occurrence graph and the DDI graph.
- **DARC (Demographic-Aware Representation Calibration)** — gated residual correction that adaptively adjusts the patient representation with age / gender information.
- **HMR (Historical Medication Retrieval)** — attention over historical visits with a context gate, plus a DDI-aware training loss.

## Code Structure

```bash
DRIMed-main/
├── src/
│   ├── Baselines/            # baseline models (CausalMed, COGNet, GAMENet, HyperMed, LEAP, MoleRec, MrDTR, RAVSNet, RETAIN, SafeDrug)
│   ├── models/
│   │   ├── DRIMed.py         # main model
│   │   └── DRIMed_wo_*.py    # ablation variants
│   └── utils/
│       ├── data_loader.py    # dataset loading
│       ├── encoder.py        # SAMFormer-based intra-visit encoder
│       ├── layers.py         # GCN layers
│       └── metrics.py        # Jaccard / F1 / PRAUC / DDI-rate metrics
├── data/output/              # preprocessed MIMIC-III tensors used by this repo
├── main_DRIMed.py            # unified training / testing entry point
├── install.sh
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch >= 1.13 (GPU recommended)

```bash
bash install.sh          # conda environment + dependencies
# or
pip install -r requirements.txt
```

## Usage

All models (main model, ablations, baselines) share one entry point:

```bash
# train the main model
python main_DRIMed.py --model DRIMed

# train an ablation variant
python main_DRIMed.py --model DRIMed_wo_DARC

# train a baseline
python main_DRIMed.py --model GAMENet

# test with a checkpoint
python main_DRIMed.py --model DRIMed --test --resume Epoch_72_TARGET_0.06_JA_0.5422_DDI_0.06674.model
```

Key arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model` | any name registered in `MODEL_REGISTRY` (`DRIMed`, ablations, baselines) | `DRIMed` |
| `--dataset` | `mimic3` or `mimic4` | `mimic3` |
| `--emb_dim` | embedding dimension | `64` |
| `--epochs` | number of training epochs | `100` |
| `--lr` | learning rate | `1e-4` |
| `--target_ddi` | target DDI rate | `0.06` |
| `--threshold` | prediction threshold | `0.5` |

Checkpoints are saved under `saved/`.

## Data

The repository ships the preprocessed files required by `src/utils/data_loader.py` under `data/output/`:

```
data/output/records_final_hhh.pkl   # patient admission records with demographic features
data/output/voc_final.pkl           # diagnosis / procedure / medication vocabularies
data/output/ehr_adj_final.pkl       # EHR co-occurrence adjacency matrix
data/output/ddi_A_final.pkl         # drug-drug interaction adjacency matrix
```

Each patient is a list of visits; each visit is `[diag_codes, proc_codes, med_codes, age, gender]`.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{drimed2026,
  title   = {DRIMed: Demographic-Aware Dynamic Clinical State Network for Medication Recommendation},
  author  = {Anonymous},
  journal = {},
  year    = {2026}
}
```
