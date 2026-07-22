import copy
import time
import argparse
import torch
import glob
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils import functions as uf
from utils import unlearning as uu
from utils import eval as ue
from utils.model import DynamicMLP
from utils.submission import save_submission, log_submission
    


# --- Setup & Decleration ---
parser = argparse.ArgumentParser()
parser.add_argument("--submit", action="store_true",
                    help="write the submission folder (TIMidi_V*)")
parser.add_argument("--method", default="finetune",
                    choices=["finetune", "gradasc", "ssd", "fisher"],
                    help="unlearning method to run")
parser.add_argument("--alpha", type=float, default=5.0, help="SSD selection threshold")
parser.add_argument("--lam", type=float, default=2.0, help="SSD dampening strength")
parser.add_argument("--sigma", type=float, default=1e-6, help="Fisher forgetting noise scale")
parser.add_argument("--fisher-eps", type=float, default=1e-4, help="Fisher forgetting stabilizer")
parser.add_argument("--ft-epochs", type=int, default=5, help="fine-tune epochs")
parser.add_argument("--ft-lr", type=float, default=1e-2, help="fine-tune learning rate")
parser.add_argument("--ft-opt", default="sgd", choices=["sgd", "adam"], help="fine-tune optimizer")
parser.add_argument("--ft-batch", type=int, default=256, help="fine-tune batch size")
parser.add_argument("--ft-subsample", type=float, default=1.0,
                    help="fraction of zero-label users to keep for fine-tune (speedup)")
parser.add_argument("--ft-sched", default="none", choices=["none", "cosine"],
                    help="fine-tune learning-rate schedule")
parser.add_argument("--ft-loss", default="bce", choices=["bce", "focal"], help="fine-tune loss")
parser.add_argument("--ft-gamma", type=float, default=2.0, help="focal loss focusing parameter")
parser.add_argument("--ga-ascent-epochs", type=int, default=1, help="gradasc: ascent epochs on Df")
parser.add_argument("--ga-ascent-lr", type=float, default=1e-3, help="gradasc: ascent lr (high = aggressive)")
parser.add_argument("--ga-repair-epochs", type=int, default=2, help="gradasc: repair epochs on Dr")
parser.add_argument("--ga-repair-lr", type=float, default=1e-2, help="gradasc: repair lr")
parser.add_argument("--ga-batch", type=int, default=256, help="gradasc: batch size")
parser.add_argument("--ga-repair-opt", default="adam", choices=["sgd", "adam"], help="gradasc: repair optimizer")
args = parser.parse_args()
np.random.seed(42)

folder_path = './data/'
artifact_path = Path('data') / 'model_artifact'
forget_path = Path(folder_path)/'forget_data.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


csv_files = glob.glob(os.path.join(folder_path, '*c000.csv'))
df_all = pd.concat((pd.read_csv(file, sep=";") for file in csv_files), ignore_index=True)

random_seed = 42
torch.manual_seed(random_seed)
id_col = "user_id"



# --- Data split & df init --- 
forget_ids = pd.read_csv(forget_path)[id_col]

forget_df = df_all[df_all[id_col].isin(forget_ids)].reset_index(drop=True)
retain_df = df_all[~df_all[id_col].isin(forget_ids)].reset_index(drop=True)

# 85% train / 15% val  (val = held-out for local eval + declared validation_ids)
train_df, val_df = train_test_split(retain_df, test_size=0.15, random_state=random_seed)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("\n--- Lengths of Dataframes")
print(f"Retain set: {len(retain_df)} \nForget set: {len(forget_df)}\n")
print(f"Train: {len(train_df)} \nVal: {len(val_df)}\n")



# --- Prepare train features & class weights ---
X_train, y_train, feature_cols, target_cols = uf.prepare_data(train_df, id_col=id_col, target_prefix='target__')

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train).astype(np.float32)


pos_counts = np.sum(y_train, axis=0)
neg_counts = len(y_train) - pos_counts
pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), device=device)
pos_weights = pos_weights.clamp(min=0.1, max=100.0)
print(f"pos_weights: {pos_weights}")



# --- Load Model from Artifact --- 
payload = uf.load_pickle(artifact_path)

state_dict = payload['state_dict']
architecture = payload['architecture']
best_params = payload['best_hyperparameters']
model_class_source = payload['model_class_source']

print("\n--- Saved Metadata ---")
print("Architecture parameters:", architecture)
print("Best Hyperparameters:", best_params)

try:
    model = DynamicMLP(
        input_dim=architecture['input_dim'],
        hidden_layers=architecture['hidden_layers'],
        num_outputs=architecture['num_outputs']
    )
except NameError:
    print("DynamicMLP class was not found. Check if the class source compiled correctly.")
    raise

model.load_state_dict(state_dict)
model.to(device)
print(sum(p.numel() for p in model.parameters()))
model.eval()

print("\nModel successfully reconstructed and weights loaded.")



# --- Prepare val / test / forget features (transform only: no leakage) ---
X_val, y_val, _, _ = uf.prepare_data(val_df, id_col=id_col, target_prefix='target__')
X_val = imputer.transform(X_val).astype(np.float32)

X_forget, y_forget, _, _ = uf.prepare_data(forget_df, id_col=id_col, target_prefix='target__')
X_forget = imputer.transform(X_forget).astype(np.float32)


# --- Baseline unlearning: fine-tune on retain set only (timed) ---
unlearned_model = copy.deepcopy(model)
start = time.perf_counter()
if args.method == "finetune":
    X_ft, y_ft = X_train, y_train
    if args.ft_subsample < 1.0:
        has_label = y_train.sum(axis=1) > 0                      # keep every informative user
        keep = has_label | (np.random.rand(len(y_train)) < args.ft_subsample)
        X_ft, y_ft = X_train[keep], y_train[keep]
        print(f"fine-tune on {len(X_ft)}/{len(X_train)} rows (subsample={args.ft_subsample})")
    unlearned_model = uu.fine_tune(unlearned_model, X_ft, y_ft, pos_weights, device,
                                   epochs=args.ft_epochs, lr=args.ft_lr,
                                   batch_size=args.ft_batch, optimizer=args.ft_opt,
                                   sched=args.ft_sched, loss_type=args.ft_loss, gamma=args.ft_gamma)
elif args.method == "gradasc":
    unlearned_model = uu.gradient_ascent(unlearned_model, X_forget, y_forget,
                                         X_train, y_train, pos_weights, device,
                                         ascent_epochs=args.ga_ascent_epochs, ascent_lr=args.ga_ascent_lr,
                                         repair_epochs=args.ga_repair_epochs, repair_lr=args.ga_repair_lr,
                                         batch_size=args.ga_batch, repair_opt=args.ga_repair_opt)
elif args.method == "ssd":
    unlearned_model = uu.ssd_unlearn(unlearned_model, X_forget, y_forget,
                                     X_train, y_train, pos_weights, device,
                                     alpha=args.alpha, lam=args.lam)
elif args.method == "fisher":
    unlearned_model = uu.fisher_forget(unlearned_model, X_train, y_train, pos_weights, device,
                                       sigma=args.sigma, eps=args.fisher_eps)
elapsed = time.perf_counter() - start


# --- Local evaluation (test = never-seen -> MIA non-member) ---
p10 = ue.precision_at_k(unlearned_model, X_val, y_val)
auc = ue.mia_auc(unlearned_model, X_forget, y_forget, X_val, y_val)
mia_score = 1 - 2 * abs(auc - 0.5)
print("\n--- Local Score ---")
print(f"Precision@10: {p10:.4f}")
print(f"MIA AUC: {auc:.4f}  (mia_score {mia_score:.4f})")
print(f"Unlearning time: {elapsed:.1f}s")


# --- Submission (only with --submit) ---
if args.submit:
    out = save_submission(unlearned_model, val_df, architecture, best_params, elapsed)
    if args.method == "ssd":
        params = f"alpha={args.alpha}, lam={args.lam}"
    elif args.method == "gradasc":
        params = (f"asc_ep={args.ga_ascent_epochs}, asc_lr={args.ga_ascent_lr}, "
                  f"rep_ep={args.ga_repair_epochs}, rep_lr={args.ga_repair_lr}")
    elif args.method == "fisher":
        params = f"sigma={args.sigma}, eps={args.fisher_eps}"
    else:
        params = (f"opt={args.ft_opt}, epochs={args.ft_epochs}, lr={args.ft_lr}, "
                  f"batch={args.ft_batch}, sub={args.ft_subsample}, sched={args.ft_sched}, "
                  f"loss={args.ft_loss}" + (f"(g={args.ft_gamma})" if args.ft_loss == "focal" else ""))
    log_submission(out, args.method, params, p10, mia_score, elapsed)
else:
    print("\n(run with --submit to write the submission folder)")