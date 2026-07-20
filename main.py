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
    


# --- Setup & Decleration --- 
folder_path = './data/'
artifact_path = Path('data') / 'model_artifact'
forget_path = Path(folder_path)/'forget_data.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


csv_files = glob.glob(os.path.join(folder_path, '*c000.csv'))
df_all = pd.concat((pd.read_csv(file, sep=";") for file in csv_files), ignore_index=True)

random_seed = 42
id_col = "user_id"



# --- Data split & df init --- 
forget_ids = pd.read_csv(forget_path)[id_col]

forget_df = df_all[df_all[id_col].isin(forget_ids)].reset_index(drop=True)
retain_df = df_all[~df_all[id_col].isin(forget_ids)].reset_index(drop=True)

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
print(sum(p.numel() for p in model.parameters()))
model.eval()

print("\nModel successfully reconstructed and weights loaded.")