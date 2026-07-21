"""K-fold cross-validation to pick unlearning hyperparameters.
The forget set is fixed; only the retain set is split into folds.
Run (from the repo root):  python -m utils.tune --method ssd
                            python -m utils.tune --method fisher
"""
import argparse
import glob
import itertools
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from pathlib import Path

from . import functions as uf
from . import unlearning as uu
from . import eval as ue
from .model import DynamicMLP

device = torch.device('cpu')
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True, choices=["ssd", "fisher"])
parser.add_argument("--folds", type=int, default=4)
args = parser.parse_args()

# One grid + one runner per method. The runner always receives the same
# arguments (forget set + one retain fold + pos_weights) so the CV loop
# below doesn't need to know which method it's driving.
GRIDS = {
    "ssd": {"alpha": [1, 2, 5, 10], "lam": [1, 2, 4]},
    "fisher": {"sigma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "eps": [1e-6, 1e-5, 1e-4, 1e-3]},
}
RUNNERS = {
    "ssd": lambda m, Xf, yf, Xr, yr, pw, **p: uu.ssd_unlearn(m, Xf, yf, Xr, yr, pw, device, **p),
    "fisher": lambda m, Xf, yf, Xr, yr, pw, **p: uu.fisher_forget(m, Xr, yr, pw, device, **p),
}

# --- data ---
csv = glob.glob('./data/*c000.csv')
df = pd.concat((pd.read_csv(f, sep=';') for f in csv), ignore_index=True)
fids = pd.read_csv('data/forget_data.csv')['user_id']
forget_df = df[df['user_id'].isin(fids)].reset_index(drop=True)
retain_df = df[~df['user_id'].isin(fids)].reset_index(drop=True)

# impute (fit once on the full retain, for simplicity)
imp = SimpleImputer(strategy='median')
Xr, yr, _, _ = uf.prepare_data(retain_df, id_col='user_id', target_prefix='target__')
Xr = imp.fit_transform(Xr).astype(np.float32)
Xf, yf, _, _ = uf.prepare_data(forget_df, id_col='user_id', target_prefix='target__')
Xf = imp.transform(Xf).astype(np.float32)

pc = yr.sum(0)
pw = torch.tensor((len(yr) - pc) / (pc + 1e-6)).clamp(0.1, 100.).float()

# pretrained model factory
payload = uf.load_pickle(Path('data') / 'model_artifact')
arch = payload['architecture']

def fresh_model():
    m = DynamicMLP(arch['input_dim'], arch['hidden_layers'], arch['num_outputs'])
    m.load_state_dict(payload['state_dict'])
    return m

# --- CV grid search ---
grid = GRIDS[args.method]
run = RUNNERS[args.method]
param_names = list(grid.keys())
combos = list(itertools.product(*grid.values()))

kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

header = "".join(f"{name:>10}" for name in param_names) + f"{'P@10':>9}{'±':>8}{'MIA':>9}"
print(f"\n{header}")
results = []
for combo in combos:
    params = dict(zip(param_names, combo))
    p10s, mias = [], []
    for tr_idx, ev_idx in kf.split(Xr):
        m = fresh_model()
        run(m, Xf, yf, Xr[tr_idx], yr[tr_idx], pw, **params)
        p10s.append(ue.precision_at_k(m, Xr[ev_idx], yr[ev_idx]))
        mias.append(1 - 2 * abs(ue.mia_auc(m, Xf, yf, Xr[ev_idx], yr[ev_idx]) - 0.5))
    mean_p10, std_p10, mean_mia = np.mean(p10s), np.std(p10s), np.mean(mias)
    results.append((params, mean_p10, std_p10, mean_mia))

    row = "".join(f"{v:>10.1e}" if isinstance(v, float) else f"{v:>10}" for v in combo)
    print(f"{row}{mean_p10:>9.4f}{std_p10:>8.4f}{mean_mia:>9.4f}")

# best P@10 among candidates where MIA is still saturated (forgetting looks trivial for everyone
# here since Df is a random subsample of the population, so MIA is a gate, not the objective)
valid = [r for r in results if r[3] >= 0.98]
best = max(valid, key=lambda r: r[1]) if valid else max(results, key=lambda r: r[1])
print(f"\nBEST: {best[0]}  |  P@10={best[1]:.4f} (±{best[2]:.4f})  MIA={best[3]:.4f}")
