
# %%
import glob
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils import functions as uf
from utils import eval as ue
from utils.model import DynamicMLP

torch.manual_seed(42)
df = pd.concat((pd.read_csv(f, sep=';') for f in glob.glob('./data/*c000.csv')), ignore_index=True)
fids = pd.read_csv('data/forget_data.csv')['user_id']
forget_df = df[df['user_id'].isin(fids)].reset_index(drop=True)
retain_df = df[~df['user_id'].isin(fids)].reset_index(drop=True)
train_df, val_df = train_test_split(retain_df, test_size=0.15, random_state=42)

Xtr, ytr, feat, targ = uf.prepare_data(train_df, id_col='user_id', target_prefix='target__')
Xva, yva, _, _ = uf.prepare_data(val_df, id_col='user_id', target_prefix='target__')
imp = SimpleImputer(strategy='median')
Xtr = imp.fit_transform(Xtr).astype(np.float32)
Xva = imp.transform(Xva).astype(np.float32)
print("train:", Xtr.shape, " val:", Xva.shape)

# %% [markdown]
# ## 1) Feature scales
# Are features comparable, or wildly different in magnitude?

# %%
stats = pd.DataFrame({
    "mean": Xtr.mean(0), "std": Xtr.std(0),
    "min": Xtr.min(0), "max": Xtr.max(0),
}, index=feat)
print(f"features with std>10: {(stats['std']>10).sum()}/{len(stats)}")
plt.figure(figsize=(7, 3))
plt.hist(np.log10(stats["std"] + 1), bins=50)
plt.xlabel("log10(feature std + 1)"); plt.ylabel("# features")
plt.title("Feature scales span many orders of magnitude"); plt.tight_layout(); plt.show()
stats.sort_values("std", ascending=False).head(8)   # -> table of the most extreme features

# %% [markdown]
# **Osservazione:** scale estreme (std fino a ~1e10). Sembra servire lo scaling,
# ma il modello dato le vuole NON scalate (vedi sezione 4).

# %% [markdown]
# ## 2) Target distribution
# How many labels per user? How imbalanced are the 28 targets?

# %%
ppu = ytr.sum(1)
print(f"avg labels/user: {ppu.mean():.2f} | users with 0 labels: {(ppu==0).mean()*100:.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].hist(ppu, bins=range(0, 12)); ax[0].set_title("Positive labels per user"); ax[0].set_xlabel("# labels")
freq = pd.Series(ytr.mean(0), index=[t.replace('target__', '') for t in targ]).sort_values()
freq.plot.barh(ax=ax[1]); ax[1].set_title("Target frequency (share of users)")
plt.tight_layout(); plt.show()
freq.to_frame("frequency").sort_values("frequency", ascending=False)   # -> table

# %% [markdown]
# **Osservazione:** ~88% degli utenti ha ZERO label; target rari (roaming,
# assicurazioni) allo 0.1%. Problema iper-sparso e sbilanciato. La P@10 conta
# solo sul ~12% con almeno una label.

# %% [markdown]
# ## 3) Forget vs Retain — il forget è casuale?

# %%
Xf, yf, _, _ = uf.prepare_data(forget_df, id_col='user_id', target_prefix='target__')
Xf = imp.transform(Xf).astype(np.float32)
diff = np.abs((Xf.mean(0) - Xtr.mean(0)) / (Xtr.std(0) + 1e-9))
print(f"max standardized mean-diff: {diff.max():.3f} | features differing >0.1 std: {(diff>0.1).sum()}/{len(diff)}")
pd.DataFrame({"retain": [ppu.mean()], "forget": [yf.sum(1).mean()]}, index=["avg labels/user"])

# %% [markdown]
# **Osservazione:** forget ~ identico al retain -> forget CASUALE -> MIA ~0.5
# gratis per tutti -> la gara e' 100% precision.

# %% [markdown]
# ## 4) Soffitto del modello dato (nessun unlearning)

# %%
p = uf.load_pickle(Path('data') / 'model_artifact')
arch = p['architecture']
m = DynamicMLP(arch['input_dim'], arch['hidden_layers'], arch['num_outputs'])
m.load_state_dict(p['state_dict']); m.eval()
sc = StandardScaler().fit(Xtr)
res = pd.DataFrame({
    "P@10 (val)": [
        ue.precision_at_k(m, Xva, yva),
        ue.precision_at_k(m, sc.transform(Xva).astype(np.float32), yva),
    ]
}, index=["unscaled (current)", "standard-scaled"])
res.plot.bar(legend=False, figsize=(5, 3), title="Given model P@10: unscaled vs scaled")
plt.tight_layout(); plt.show()
res

# %% [markdown]
# **Osservazione:** unscaled 0.64, scaled 0.34 -> il modello VUOLE input non
# scalati. 0.64 e' il suo soffitto; SSD arriva a ~0.636 (= il modello originale).
# Nessun unlearning del modello dato puo' superare ~0.64 -> per andare oltre
# serve un modello MIGLIORE (retrain).
