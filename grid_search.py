"""Grid search fine-tune hyperparameters against local true-label P@10
(which tracks the leaderboard precision with a ~+0.035 offset).
Run:  python grid_search.py
"""
import glob, time, itertools
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils import functions as uf
from utils import unlearning as uu
from utils import eval as ue
from utils.model import DynamicMLP

device = torch.device('cpu')
torch.manual_seed(42); np.random.seed(42)

df = pd.concat((pd.read_csv(f, sep=';') for f in glob.glob('./data/*c000.csv')), ignore_index=True)
fids = pd.read_csv('data/forget_data.csv')['user_id']
retain_df = df[~df['user_id'].isin(fids)].reset_index(drop=True)
train_df, val_df = train_test_split(retain_df, test_size=0.15, random_state=42)

Xtr, ytr, _, _ = uf.prepare_data(train_df, id_col='user_id', target_prefix='target__')
imp = SimpleImputer(strategy='median')
Xtr = imp.fit_transform(Xtr).astype(np.float32)
Xva, yva, _, _ = uf.prepare_data(val_df, id_col='user_id', target_prefix='target__')
Xva = imp.transform(Xva).astype(np.float32)
pc = ytr.sum(0)
pw = torch.tensor((len(ytr) - pc) / (pc + 1e-6)).clamp(0.1, 100.).float()

payload = uf.load_pickle(Path('data') / 'model_artifact')
arch = payload['architecture']
def given():
    m = DynamicMLP(arch['input_dim'], arch['hidden_layers'], arch['num_outputs'])
    m.load_state_dict(payload['state_dict']); return m

# --- grid (now includes batch) ---
EPOCHS = [10, 20]
LR = [5e-4, 1e-3]
SUB = [0.1, 0.3]
BATCH = [256, 512, 1024]

print(f"\n{'ep':>4}{'lr':>9}{'sub':>6}{'batch':>7}{'p10':>8}{'time':>7}")
rows = []
for ep, lr, sub, bs in itertools.product(EPOCHS, LR, SUB, BATCH):
    m = given()
    X_ft, y_ft = Xtr, ytr
    if sub < 1.0:
        keep = (ytr.sum(1) > 0) | (np.random.rand(len(ytr)) < sub)
        X_ft, y_ft = Xtr[keep], ytr[keep]
    t0 = time.perf_counter()
    uu.fine_tune(m, X_ft, y_ft, pw, device, epochs=ep, lr=lr, batch_size=bs, optimizer='adam')
    el = time.perf_counter() - t0
    p10 = ue.precision_at_k(m, Xva, yva)
    rows.append((ep, lr, sub, bs, p10, el))
    print(f"{ep:>4}{lr:>9.0e}{sub:>6}{bs:>7}{p10:>8.4f}{el:>7.1f}", flush=True)

best = max(rows, key=lambda r: r[4])
print(f"\nBEST p10: ep={best[0]} lr={best[1]:.0e} sub={best[2]} batch={best[3]} "
      f"-> p10={best[4]:.4f} (~{best[4]+0.035:.3f} leaderboard, {best[5]:.1f}s)")
