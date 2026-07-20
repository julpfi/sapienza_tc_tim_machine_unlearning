import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def precision_at_k(model, X, y, k=10, batch_size=4096):
    # Fraction of each user's true labels that appear in the top-k
    # predicted classes, averaged over users with at least one label.
    model.eval()
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            probs_list.append(model.predict_proba(X[i:i + batch_size]))
    probs = np.vstack(probs_list)

    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    topk_mask = np.zeros_like(y, dtype=bool)
    topk_mask[np.arange(len(y))[:, None], topk_idx] = True

    n_pos = y.sum(axis=1)
    hits = (y.astype(bool) & topk_mask).sum(axis=1)

    valid = n_pos > 0  # skip users with no positive labels
    return (hits[valid] / n_pos[valid]).mean()



def per_sample_bce(model, X, y, batch_size=4096):
    # Mean BCE loss per user, computed on logits for numerical stability.
    model.eval()
    losses = []
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            yb = torch.tensor(y[i:i + batch_size], dtype=torch.float32)
            logits = model(xb)
            losses.append(loss_fn(logits, yb).mean(dim=1).numpy())
    return np.concatenate(losses)


def mia_auc(model, X_forget, y_forget, X_nonmember, y_nonmember):
    # Loss-based membership inference: can an attacker separate forget
    # samples from non-members using the per-sample loss?
    # AUC ~ 0.5 means forgetting worked, ~1.0 means it did not.
    loss_f = per_sample_bce(model, X_forget, y_forget)
    loss_n = per_sample_bce(model, X_nonmember, y_nonmember)

    labels = np.concatenate([np.ones(len(loss_f)), np.zeros(len(loss_n))])
    # lower loss -> more likely member, so negate the score
    scores = -np.concatenate([loss_f, loss_n])
    return roc_auc_score(labels, scores)
