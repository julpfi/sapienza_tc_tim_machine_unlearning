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
