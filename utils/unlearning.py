import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _get_data_loader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fine_tune(model, X_retain, y_retain, pos_weights, device, epochs=5, lr=1e-2, batch_size=256):
    """
    Baseline unlearning - continue SGD on the retain set Dr for a few epochs
    """
    print("\nExecute fine_tune()")
    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device=device, dtype=torch.float32))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = _get_data_loader(X_retain, y_retain, batch_size)

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch + 1}/{epochs} - avg retain loss: {running_loss / n_batches:.4f}")

    model.eval()
    return model


def gradient_ascent(model, X_forget, y_forget, X_retain, y_retain, pos_weights, device,
                     ascent_epochs=1, ascent_lr=1e-3, repair_epochs=2, repair_lr=1e-2, batch_size=256):
    """
    NegGrad+ : a few gradient-ASCENT steps on the forget set Df (push the model
    away from fitting it), then a short fine-tune on Dr to repair utility
    """
    print("\nExecute gradient_ascent()")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device=device, dtype=torch.float32))

    # --- ascent phase: maximise loss on Df (minimise -loss) ---
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=ascent_lr)
    forget_loader = _get_data_loader(X_forget, y_forget, batch_size)
    for epoch in range(ascent_epochs):
        for xb, yb in forget_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = -criterion(model(xb), yb)  # negative -> ascent
            loss.backward()
            optimizer.step()
        print(f"ascent epoch {epoch + 1}/{ascent_epochs}")

    # --- repair phase: normal fine-tune on Dr to recover precision ---
    optimizer = torch.optim.SGD(model.parameters(), lr=repair_lr)
    retain_loader = _get_data_loader(X_retain, y_retain, batch_size)
    for epoch in range(repair_epochs):
        for xb, yb in retain_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        print(f"repair epoch {epoch + 1}/{repair_epochs}")

    model.eval()
    return model


def _fisher_diag(model, X, y, pos_weights, device, batch_size=64):
    """
    Diagonal empirical Fisher Information: mean squared gradient per parameter,
    estimated over the given dataset. Shared by every Fisher-based method below
    so SSD and Fisher forgetting always agree on what "important" means.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device=device, dtype=torch.float32))
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    loader = _get_data_loader(X, y, batch_size, shuffle=False)
    n_seen = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        model.zero_grad()
        criterion(model(xb), yb).backward()

        bs = xb.size(0)
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += (p.grad.detach() ** 2) * bs
        n_seen += bs

    for name in fisher:
        fisher[name] /= n_seen
    return fisher


def ssd_unlearn(model, X_forget, y_forget, X_retain, y_retain, pos_weights, device, alpha=1.0, lam=1.0):
    """
    Selective Synaptic Dampening (Foster et al.) - shrink only the weights that
    are important for the forget set Df but not for the retain set Dr.
    """
    print("\nExecute ssd_unlearn()")
    model.to(device)
    fisher_forget_set = _fisher_diag(model, X_forget, y_forget, pos_weights, device)
    fisher_retain_set = _fisher_diag(model, X_retain, y_retain, pos_weights, device)

    eps = 1e-12
    n_shrunk = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            ff, fr = fisher_forget_set[name], fisher_retain_set[name]
            selection = ff > alpha * fr  # forget-specific weights
            beta = torch.clamp(lam * fr / (ff + eps), max=1.0)
            factor = torch.where(selection, beta, torch.ones_like(beta))
            p.mul_(factor)
            n_shrunk += int(selection.sum())
    print(f"shrunk {n_shrunk} weights (alpha={alpha}, lam={lam})")

    model.eval()
    return model


def fisher_forget(model, X_retain, y_retain, pos_weights, device, sigma=1e-2, eps=1e-6):
    """
    Fisher forgetting - keep the weights that matter for the retain set Dr untouched, and perturb the rest with noise
    That's where information about the forgotten data Df is assumed to live: 
        High Fisher (important for Dr)  -> small variance -> little noise   
        Low Fisher (unimportant for Dr) -> large variance -> more noise
    """
    print("\nExecute fisher_forget()")
    model.to(device)
    fisher_retain_set = _fisher_diag(model, X_retain, y_retain, pos_weights, device)

    with torch.no_grad():
        for name, p in model.named_parameters():
            std = (sigma ** 2 / (fisher_retain_set[name] + eps)).sqrt()
            p.add_(torch.randn_like(p) * std)

    model.eval()
    return model
