import torch
import glob
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils import functions as uf
from utils.model import DynamicMLP
from utils.eval import precision_at_k, mia_auc

folder_path = './data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


csv_files = glob.glob(os.path.join(folder_path, '*c000.csv'))
df_all = pd.concat((pd.read_csv(file, sep=";") for file in csv_files), ignore_index=True)

random_seed = 42
id_col = "user_id"

forget_path = Path(folder_path)/'forget_data.csv'
forget_ids = pd.read_csv(forget_path)[id_col]

forget_df = df_all[df_all[id_col].isin(forget_ids)].reset_index(drop=True)
retain_df = df_all[~df_all[id_col].isin(forget_ids)].reset_index(drop=True)

train_df, val_df = train_test_split(retain_df, test_size=0.15, random_state=random_seed)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Retain pool: {len(retain_df)} \nForget set: {len(forget_df)}\n")
print(f"Train: {len(train_df)} \nVal: {len(val_df)}\n")


X_train, y_train, feature_cols, target_cols = uf.prepare_data(train_df, id_col=id_col, target_prefix='target__')

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train).astype(np.float32)



pos_counts = np.sum(y_train, axis=0)
neg_counts = len(y_train) - pos_counts
pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), device=device)
pos_weights = pos_weights.clamp(min=0.1, max=100.0)
print(f"pos_weights: {pos_weights}")


artifact_path = Path('data') / 'model_artifact'

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

model.eval()

print("\nModel successfully reconstructed and weights loaded.")


# evaluation: imputer fitted on train, transform only here
X_val, y_val, _, _ = uf.prepare_data(val_df, id_col=id_col, target_prefix='target__')
X_val = imputer.transform(X_val).astype(np.float32)

X_forget, y_forget, _, _ = uf.prepare_data(forget_df, id_col=id_col, target_prefix='target__')
X_forget = imputer.transform(X_forget).astype(np.float32)

p10 = precision_at_k(model, X_val, y_val, k=10)
print(f"\nPrecision@10 (validation): {p10:.4f}")

auc = mia_auc(model, X_forget, y_forget, X_val, y_val)
mia_score = 1.0 - 2.0 * abs(auc - 0.5)
print(f"MIA AUC (forget vs val): {auc:.4f}  ->  MIA score: {mia_score:.4f}")