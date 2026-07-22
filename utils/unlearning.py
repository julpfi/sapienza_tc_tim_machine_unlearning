import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _get_data_loader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _layer_of(param_name):
    """'net.3.weight' -> 'net.3' - the module a parameter belongs to."""
    return ".".join(param_name.split(".")[:2])


def _selected(param_name, layers):
    """layers=None means unrestricted (every parameter); otherwise whitelist by module prefix."""
    return layers is None or _layer_of(param_name) in layers


def fine_tune(model, X_retain, y_retain, pos_weights, device, epochs=5, lr=1e-2,
              batch_size=256, optimizer="sgd"):
    """
    Fine-tune on the retain set Dr. Heavier training pushes the model toward
    A(Dr) (the retrain-on-retain reference). Use optimizer="adam" + more epochs
    for the "invasive" variant.
    """
    print(f"\nExecute fine_tune() [opt={optimizer}, epochs={epochs}, lr={lr}]")
    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device=device, dtype=torch.float32))
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
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


def _fisher_diag(model, X, y, pos_weights, device, batch_size=64, layers=None):
    """
    Diagonal empirical Fisher Information: mean squared gradient per parameter,
    estimated over the given dataset. Shared by every Fisher-based method below
    so SSD and Fisher forgetting always agree on what "important" means.

    layers restricts which parameters are tracked (e.g. ("net.3",) to only look
    at the second Linear layer). Default None means every parameter.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device=device, dtype=torch.float32))
    tracked = [(name, p) for name, p in model.named_parameters() if _selected(name, layers)]
    fisher = {name: torch.zeros_like(p) for name, p in tracked}

    loader = _get_data_loader(X, y, batch_size, shuffle=False)
    n_seen = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        model.zero_grad()
        criterion(model(xb), yb).backward()

        bs = xb.size(0)
        for name, p in tracked:
            if p.grad is not None:
                fisher[name] += (p.grad.detach() ** 2) * bs
        n_seen += bs

    for name in fisher:
        fisher[name] /= n_seen
    return fisher


def ssd_unlearn(model, X_forget, y_forget, X_retain, y_retain, pos_weights, device,
                 alpha=1.0, lam=1.0, layers=("net.3",)):
    """
    Selective Synaptic Dampening (Foster et al.) - shrink only the weights that
    are important for the forget set Df but not for the retain set Dr.

    layers whitelists which modules are touched at all (default: only the
    second Linear layer). net[0] takes raw, unstandardized features, so its
    Fisher values are unreliable and dampening it is destructive; BatchNorm
    modules are never included either since they only have module prefixes
    net.1/net.4 (net.2/net.5 are ReLU, no parameters, listed for clarity).
    """
    print("\nExecute ssd_unlearn()")
    model.to(device)
    fisher_forget_set = _fisher_diag(model, X_forget, y_forget, pos_weights, device, layers=layers)
    fisher_retain_set = _fisher_diag(model, X_retain, y_retain, pos_weights, device, layers=layers)

    eps = 1e-12
    n_shrunk = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in fisher_forget_set:
                continue
            ff, fr = fisher_forget_set[name], fisher_retain_set[name]
            selection = ff > alpha * fr  # forget-specific weights
            beta = torch.clamp(lam * fr / (ff + eps), max=1.0)
            factor = torch.where(selection, beta, torch.ones_like(beta))
            p.mul_(factor)
            n_shrunk += int(selection.sum())
    print(f"shrunk {n_shrunk} weights (alpha={alpha}, lam={lam}, layers={layers})")

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


def recalibrate_head(model, X_retain, y_retain, device, epochs=300, lr=0.1):
    """
    Per-class affine recalibration of the frozen head: z'_c = a_c * z_c + beta_c 
    Only these two scalars are fit (starting with identity i.e. a=1, beta=0),
    The fit is then folded straight into net[6].weight/bias
    """
    print("\nExecute recalibrate_head()")
    model.to(device)
    model.eval()

    with torch.no_grad():
        X = torch.as_tensor(X_retain, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_retain, dtype=torch.float32, device=device)
        logits = model(X)  # frozen forward pass through the untouched head

    num_classes = logits.shape[1]
    a = torch.ones(num_classes, device=device, requires_grad=True)
    beta = torch.zeros(num_classes, device=device, requires_grad=True)

    criterion = nn.BCEWithLogitsLoss()  # unweighted, per the spec
    optimizer = torch.optim.Adam([a, beta], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(a * logits + beta, y)
        loss.backward()
        optimizer.step()
    print(f"recalibration loss: {loss.item():.4f}")

    head = model.net[6]
    with torch.no_grad():
        head.weight.mul_(a.detach().unsqueeze(1))
        head.bias.mul_(a.detach()).add_(beta.detach())

    model.eval()
    return model
