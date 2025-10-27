from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def evaluate(model: torch.nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    writer: Optional["SummaryWriter"] = None,
    device: Optional[torch.device] = None,
    log_hist_every: int = 100,
):
    device = device or torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    global_step = 0
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred, z = model(xb)

            # Basic shape check to catch target_fn/output mismatch early
            if pred.shape != yb.shape:
                raise RuntimeError(
                    f"Prediction shape {tuple(pred.shape)} does not match target shape {tuple(yb.shape)}."
                )

            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)
            batch_count += xb.size(0)
            global_step += 1

            if writer is not None:
                writer.add_scalar("train/batch_loss", loss.item(), global_step)
                if log_hist_every and (global_step % log_hist_every == 0):
                    # flatten embeddings and log a histogram
                    writer.add_histogram("embed/batch", z.detach().cpu().numpy(), global_step)

        epoch_loss = running_loss / max(batch_count, 1)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        best_val = min(best_val, val_loss)

        if writer is not None:
            writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
            writer.add_scalar("val/epoch_loss", val_loss, epoch)
            for i, param_group in enumerate(opt.param_groups):
                writer.add_scalar(f"lr/group_{i}", param_group.get("lr", 0.0), epoch)

        if (epoch % 5 == 0) or (epoch == 1) or (epoch == epochs):
            print(f"Epoch {epoch:02d}/{epochs} | val MSE: {val_loss:.6f}")

    return {"best_val": best_val, "last_val": val_loss}
