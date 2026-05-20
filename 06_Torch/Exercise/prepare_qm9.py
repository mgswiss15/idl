"""Prepare QM9 dataset as tabular CSV with molecular fingerprints.

This script:
  1. Downloads QM9 via PyTorch Geometric (requires torch_geometric)
  2. Computes Morgan fingerprints (1024 bits, radius 2) from SMILES
  3. Saves train.csv, test.csv, and sample_submission.csv

Targets predicted (multi-output regression):
  - homo:  HOMO energy (eV)
  - lumo:  LUMO energy (eV)
  - gap:   HOMO-LUMO gap (eV)
  - mu:    dipole moment (Debye)
  - alpha: isotropic polarisability (Bohr^3)

Usage:
  pip install torch torch_geometric rdkit
  python prepare_qm9.py

Output files:
  qm9_train.csv   -- 110 000 molecules, features + targets
  qm9_test.csv    -- 24 831 molecules,  features only
  qm9_targets.csv -- held-out targets for evaluation (do not release)
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.datasets import QM9

# ── config ────────────────────────────────────────────────────────────────────

DATA_DIR    = './qm9_raw'       # where PyG downloads raw data
OUT_DIR     = '.'               # where CSVs are saved
N_BITS      = 1024              # Morgan fingerprint length
RADIUS      = 2                 # Morgan fingerprint radius
TRAIN_SIZE  = 110_000
RANDOM_SEED = 42

# QM9 target indices and names (from PyG QM9 documentation)
# Full list has 19 targets; we use 5 most commonly predicted
TARGET_COLS = {
    0:  'mu',     # dipole moment (D)
    1:  'alpha',  # polarisability (a0^3)
    6:  'homo',   # HOMO energy (Ha) — converted to eV below
    7:  'lumo',   # LUMO energy (Ha) — converted to eV below
    8:  'gap',    # HOMO-LUMO gap (Ha) — converted to eV below
}
HA_TO_EV = 27.2114  # Hartree to eV conversion

# ── load QM9 ─────────────────────────────────────────────────────────────────

print("Loading QM9 via PyTorch Geometric...")
dataset = QM9(root=DATA_DIR)
print(f"Total molecules: {len(dataset)}")

# ── compute Morgan fingerprints ───────────────────────────────────────────────

print("Computing Morgan fingerprints...")

def smiles_from_pyg(data):
    """Extract SMILES from a PyG QM9 data object."""
    # PyG QM9 stores SMILES in data.smiles attribute
    return data.smiles

def morgan_fp(smi, n_bits=N_BITS, radius=RADIUS):
    """Compute Morgan fingerprint as numpy array. Returns None on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)

fingerprints = []
targets      = []
valid_idx    = []

for i, data in enumerate(dataset):
    if i % 10_000 == 0:
        print(f"  {i}/{len(dataset)}")

    smi = data.smiles
    fp  = morgan_fp(smi)
    if fp is None:
        continue  # skip molecules rdkit can't parse

    fingerprints.append(fp)

    # extract targets — shape (19,) per molecule
    y = data.y.squeeze().numpy()
    targets.append({
        'mu':    float(y[0]),
        'alpha': float(y[1]),
        'homo':  float(y[6]) * HA_TO_EV,
        'lumo':  float(y[7]) * HA_TO_EV,
        'gap':   float(y[8]) * HA_TO_EV,
    })
    valid_idx.append(i)

print(f"Valid molecules: {len(fingerprints)}")

# ── assemble dataframe ────────────────────────────────────────────────────────

fp_cols = [f'fp_{i}' for i in range(N_BITS)]
df_fp   = pd.DataFrame(np.stack(fingerprints), columns=fp_cols)
df_tgt  = pd.DataFrame(targets)
df      = pd.concat([df_fp, df_tgt], axis=1)
df.insert(0, 'molecule_id', valid_idx)

print(f"DataFrame shape: {df.shape}")
print(f"Target stats:\n{df[['mu','alpha','homo','lumo','gap']].describe().round(3)}")

# ── train / test split ────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)
idx = rng.permutation(len(df))

train_idx = idx[:TRAIN_SIZE]
test_idx  = idx[TRAIN_SIZE:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(df_train)}, Test: {len(df_test)}")

# ── save ──────────────────────────────────────────────────────────────────────

target_names = ['mu', 'alpha', 'homo', 'lumo', 'gap']

# training set — features + targets
df_train.to_csv(f'{OUT_DIR}/qm9_train.csv', index=False)
print(f"Saved qm9_train.csv")

# test set — features only (no targets)
df_test[['molecule_id'] + fp_cols].to_csv(
    f'{OUT_DIR}/qm9_test.csv', index=False)
print(f"Saved qm9_test.csv")

# ground truth targets — DO NOT RELEASE to students
df_test[['molecule_id'] + target_names].to_csv(
    f'{OUT_DIR}/qm9_test_targets.csv', index=False)
print(f"Saved qm9_test_targets.csv (hold this back!)")

# sample submission
df_sample = df_test[['molecule_id']].copy()
for col in target_names:
    df_sample[col] = 0.0
df_sample.to_csv(f'{OUT_DIR}/qm9_sample_submission.csv', index=False)
print(f"Saved qm9_sample_submission.csv")

print("\nDone. Files ready to upload to Kaggle.")
print(f"Features: {N_BITS} Morgan fingerprint bits")
print(f"Targets:  {target_names}")
print(f"Train:    {len(df_train)} molecules")
print(f"Test:     {len(df_test)} molecules")
