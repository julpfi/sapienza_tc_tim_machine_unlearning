import os
import inspect
import pickle
import datetime

from utils.model import DynamicMLP


def log_submission(out_dir, method, params, p10, mia_score, elapsed,
                   log_path="submissions.md", notes=""):
    """Append one row to submissions.md (creates it with a header if missing)."""
    version = os.path.basename(out_dir.rstrip("/"))
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("# Submissions log\n\n"
                    "| Version | Date | Method | Params | P@10 | MIA score | Time(s) | Notes |\n"
                    "|---|---|---|---|---|---|---|---|\n")
    with open(log_path, "a") as f:
        f.write(f"| {version} | {date} | {method} | {params} | "
                f"{p10:.4f} | {mia_score:.4f} | {elapsed:.1f} | {notes} |\n")
    print(f"Logged {version} to {log_path}")


def next_version_dir(group="TIMidi", root="."):
    """Return the next free "<group>_V<n>" folder (auto-incrementing)."""
    n = 0
    for name in os.listdir(root):
        if name.startswith(group + "_V") and name[len(group) + 2:].isdigit():
            n = max(n, int(name[len(group) + 2:]))
    return os.path.join(root, f"{group}_V{n + 1}")


def save_submission(model, val_df, architecture, best_params, elapsed,
                    out_dir=None, group="TIMidi", id_col="user_id"):
    """Write the 3 submission files. If out_dir is None, auto-pick <group>_V<n>."""
    if out_dir is None:
        out_dir = next_version_dir(group)
    os.makedirs(out_dir, exist_ok=True)

    # execution_time.txt: single integer (seconds)
    with open(os.path.join(out_dir, "execution_time.txt"), "w") as f:
        f.write(str(int(round(elapsed))))

    # model_artifact: pickle with the 4 required keys, updated weights
    payload = {
        "state_dict": model.state_dict(),
        "architecture": architecture,
        "best_hyperparameters": best_params,
        "model_class_source": inspect.getsource(DynamicMLP),
    }
    with open(os.path.join(out_dir, "model_artifact"), "wb") as f:
        pickle.dump(payload, f)

    # validation_ids.csv: one "user_id" column
    val_df[[id_col]].to_csv(os.path.join(out_dir, "validation_ids.csv"), index=False)

    print(f"Submission written to {out_dir}/")
    return out_dir
