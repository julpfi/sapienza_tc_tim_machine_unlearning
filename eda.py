"""
Exploratory Data Analysis - TIM machine unlearning
Run:  python eda.py

Findings summary (observed on our local split):
  1. Features are on wildly different scales BUT the given model expects them
     unscaled -> scaling is NOT a usable lever (see section 4).
  2. Data is extremely sparse: ~88% of users have zero interest labels.
  3. The forget set is a RANDOM sample of the population -> the MIA is trivially
     ~0.5 for everyone -> the competition is decided ONLY by precision.
  4. The given model tops out at ~0.64 P@10 on our val. Any unlearning of the
     given model cannot beat this: 0.64 is the model's own ceiling.
"""
import glob
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils import functions as uf
from utils import eval as ue
from utils.model import DynamicMLP

torch.manual_seed(42)

# --- load data (same as main.py) ---
csv = glob.glob('./data/*c000.csv')
df = pd.concat((pd.read_csv(f, sep=';') for f in csv), ignore_index=True)
fids = pd.read_csv('data/forget_data.csv')['user_id']
forget_df = df[df['user_id'].isin(fids)].reset_index(drop=True)
retain_df = df[~df['user_id'].isin(fids)].reset_index(drop=True)
train_df, val_df = train_test_split(retain_df, test_size=0.15, random_state=42)

Xtr, ytr, feat, targ = uf.prepare_data(train_df, id_col='user_id', target_prefix='target__')
Xva, yva, _, _ = uf.prepare_data(val_df, id_col='user_id', target_prefix='target__')
imp = SimpleImputer(strategy='median')
Xtr = imp.fit_transform(Xtr).astype(np.float32)
Xva = imp.transform(Xva).astype(np.float32)


# =====================================================================
# 1) FEATURE SCALES
#    Are features comparable, or do they need normalisation?
# =====================================================================
print("=== 1) FEATURE SCALES (imputed train) ===")
maxs, means, stds = Xtr.max(0), Xtr.mean(0), Xtr.std(0)
print(f"num features: {Xtr.shape[1]}")
print(f"feature-means range: {means.min():.2f} .. {means.max():.2e}")
print(f"feature-stds  range: {stds.min():.4f} .. {stds.max():.2e}")
print(f"features with std>10 : {(stds>10).sum()}/{len(stds)}")
# OBSERVED: means up to ~2e10, stds up to ~7e10, 72/381 features with std>10.
# -> scales are extreme. Looks like it needs scaling... but see section 4:
#    the given model expects UNSCALED input, so we must NOT scale.


# =====================================================================
# 2) TARGET DISTRIBUTION
#    How many labels per user? How imbalanced are the 28 targets?
# =====================================================================
print("\n=== 2) TARGET DISTRIBUTION ===")
ppu = ytr.sum(1)
print(f"avg positive labels/user: {ppu.mean():.2f} (median {np.median(ppu):.0f}, max {int(ppu.max())})")
print(f"users with 0 labels: {(ppu==0).mean()*100:.1f}%")
freq = ytr.mean(0); order = np.argsort(-freq)
print("most common:", [(targ[i].replace('target__',''), round(float(freq[i]),3)) for i in order[:5]])
print("rarest:     ", [(targ[i].replace('target__',''), round(float(freq[i]),3)) for i in order[-5:]])
# OBSERVED: 88% of users have ZERO labels; rare targets (roaming, assicurazioni)
# at ~0.1%. -> very sparse, very imbalanced multi-label problem. P@10 is scored
# only on the ~12% of users that have at least one label (eval.py: n_pos>0).


# =====================================================================
# 3) FORGET vs RETAIN
#    Is the forget set a random sample, or a distinct subpopulation?
# =====================================================================
print("\n=== 3) FORGET vs RETAIN ===")
Xf, yf, _, _ = uf.prepare_data(forget_df, id_col='user_id', target_prefix='target__')
Xf = imp.transform(Xf).astype(np.float32)
gm = np.abs((Xf.mean(0) - Xtr.mean(0)) / (stds + 1e-9))
print(f"max standardized mean-diff: {gm.max():.3f} (feature {feat[gm.argmax()]})")
print(f"features differing >0.1 std: {(gm>0.1).sum()}/{len(gm)}")
print(f"forget positives/user: {yf.sum(1).mean():.2f}  vs retain {ppu.mean():.2f}")
# OBSERVED: forget ~ identical to retain (max diff 0.04 std, 0/381 features
# differ). -> forget is a RANDOM subset -> forgetting is trivial -> MIA ~0.5 for
# everyone -> the 45% MIA is NOT contestable, the game is 100% precision.


# =====================================================================
# 4) GIVEN MODEL CEILING
#    What precision does the ORIGINAL model reach, before any unlearning?
#    (computed locally from the provided weights + our val split)
# =====================================================================
print("\n=== 4) GIVEN MODEL precision@10 on val (NO unlearning) ===")
p = uf.load_pickle(Path('data') / 'model_artifact')
arch = p['architecture']
m = DynamicMLP(arch['input_dim'], arch['hidden_layers'], arch['num_outputs'])
m.load_state_dict(p['state_dict']); m.eval()
sc = StandardScaler().fit(Xtr)
print(f"unscaled (current pipeline): P@10 = {ue.precision_at_k(m, Xva, yva):.4f}")
print(f"standard-scaled input      : P@10 = {ue.precision_at_k(m, sc.transform(Xva).astype(np.float32), yva):.4f}")
# OBSERVED: unscaled 0.638, scaled 0.336 (crashes) -> model expects unscaled.
# 0.638 is the given model's own ceiling on our val; SSD reaches ~0.636, i.e. it
# just returns the original model. No unlearning method can beat ~0.64 on this
# model -> to go higher you need a BETTER model (retrain), rules permitting.
