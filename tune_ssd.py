"""K-fold cross-validation to pick SSD hyperparameters (alpha, lam).
The forget set is fixed; only the retain set is split into folds.
Run:  python tune_ssd.py
"""
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from pathlib import Path

from utils import functions as uf
from utils import unlearning as uu
from utils import eval as ue
from utils.model import DynamicMLP

device = torch.device('cpu')
torch.manual_seed(42)

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
GRID_ALPHA = [1, 2, 5, 10]
GRID_LAM = [1, 2, 4]
K = 4
kf = KFold(n_splits=K, shuffle=True, random_state=42)

print(f"\n{'alpha':>6}{'lam':>6}{'P@10':>9}{'±':>8}{'MIA':>9}")
results = []
for a in GRID_ALPHA:
    for l in GRID_LAM:
        p10s, mias = [], []
        for tr_idx, ev_idx in kf.split(Xr):
            m = fresh_model()
            uu.ssd_unlearn(m, Xf, yf, Xr[tr_idx], yr[tr_idx], pw, device, alpha=a, lam=l)
            p10s.append(ue.precision_at_k(m, Xr[ev_idx], yr[ev_idx]))
            mias.append(1 - 2 * abs(ue.mia_auc(m, Xf, yf, Xr[ev_idx], yr[ev_idx]) - 0.5))
        mp, sp, mm = np.mean(p10s), np.std(p10s), np.mean(mias)
        results.append((a, l, mp, sp, mm))
        print(f"{a:>6}{l:>6}{mp:>9.4f}{sp:>8.4f}{mm:>9.4f}")

# best P@10 with MIA still saturated
valid = [r for r in results if r[4] >= 0.98]
best = max(valid, key=lambda r: r[2])
print(f"\nBEST: alpha={best[0]}, lam={best[1]}  |  P@10={best[2]:.4f} (±{best[3]:.4f})  MIA={best[4]:.4f}")
