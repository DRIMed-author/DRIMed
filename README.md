# DRIMed

Official PyTorch implementation of DRIMed: demographic-aware dynamic clinical state network for safe medication recommendation on MIMIC-III.

## Folder Specification

- `data/output/` folder contains the preprocessed MIMIC-III data used by this repo.
  - `records_final_hhh.pkl`: patient records; each visit is `[diag_codes, proc_codes, med_codes, age, gender]`.
  - `voc_final.pkl`: diagnosis / procedure / medication vocabularies.
  - `ehr_adj_final.pkl`: drug co-occurrence adjacency matrix.
  - `ddi_A_final.pkl`: drug-drug interaction adjacency matrix.
- `src/` folder contains all the source code.
  - `Baselines/`: baseline models (CausalMed, COGNet, GAMENet, HyperMed, LEAP, MoleRec, MrDTR, RAVSNet, RETAIN, SafeDrug).
  - `models/`: `DRIMed.py` (main model) and `DRIMed_wo_*.py` (ablation variants).
  - `utils/`: `data_loader.py`, `encoder.py`, `layers.py`, `metrics.py`.
- `main_DRIMed.py`: train or evaluate all models (main model, ablations, baselines).

## Installation

```bash
bash install.sh
# or
pip install -r requirements.txt
```

## Run the code

```bash
# train the main model
python main_DRIMed.py --model DRIMed

# train an ablation variant
python main_DRIMed.py --model DRIMed_wo_DARC

# train a baseline
python main_DRIMed.py --model GAMENet

# test with a checkpoint
python main_DRIMed.py --model DRIMed --test --resume saved/xxx.model
```

Checkpoints are saved under `saved/`.
