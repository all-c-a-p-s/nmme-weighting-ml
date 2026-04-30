"""
Train a model from prepared ML data
"""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


class PrecipDataset(Dataset):
    def __init__(self, path):
        print("loading data...")
        data = np.load(path)
        self.X = torch.tensor(data["X_input"], dtype=torch.float32)
        self.y = torch.tensor(data["y_obs"], dtype=torch.float32)
        self.p = torch.tensor(data["preds"], dtype=torch.float32)
        print(f"loaded {len(self.y):,} samples")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.p[idx]


class Senate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 7),
        )

    def forward(self, x):
        return self.net(x)


N_QUANTILES = 3


def compute_metrics(pred, obs, months):
    mae = np.mean(np.abs(pred - obs))
    rmse = np.mean(np.sqrt(np.mean((pred - obs) ** 2, axis=0)))

    unique_months = np.unique(months)

    obs_clim = np.zeros_like(obs)

    for m in unique_months:
        mask = months == m
        obs_clim[mask] = obs[mask].mean(axis=0, keepdims=True)

    obs_anom = obs - obs_clim
    pred_anom = pred - obs_clim  # use obs climatology for both
    num = np.mean(obs_anom * pred_anom, axis=0)
    den = np.sqrt(np.mean(obs_anom**2, axis=0)) * np.sqrt(np.mean(pred_anom**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.nanmean(num / den)

    quantiles = np.linspace(0, 1, N_QUANTILES + 1)
    correct = np.zeros(obs.shape, dtype=bool)
    for m in unique_months:
        mask = months == m
        o_m = obs[mask]
        f_m = pred[mask]
        bins = np.quantile(o_m, quantiles, axis=0)
        interior = bins[1:-1]
        o_bin = (o_m[np.newaxis] > interior[:, np.newaxis]).sum(axis=0)
        f_bin = (f_m[np.newaxis] > interior[:, np.newaxis]).sum(axis=0)
        correct[mask] = o_bin == f_bin
    nile_acc = correct.mean()

    return mae, rmse, acc, nile_acc


def loaders():
    dataset = PrecipDataset("data/ml_data.npz")

    print("splitting dataset...")
    train_timesteps = 300
    test_timesteps = 57
    train_size = train_timesteps * 179 * 360
    test_size = test_timesteps * 179 * 360
    train_set = Subset(dataset, range(train_size))
    test_set = Subset(dataset, range(train_size, train_size + test_size))
    print(f"train: {train_size:,}  test: {test_size:,}")

    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)

    return train_loader, test_loader, train_timesteps, test_timesteps


def train():
    model = Senate()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    EPOCHS = 20
    print(f"\nstarting training for {EPOCHS} epochs...")
    best_loss = np.inf

    train_loader, test_loader, train_timesteps, test_timesteps = loaders()
    grid_shape = (test_timesteps, 179, 360)
    test_months = np.arange(train_timesteps, train_timesteps + test_timesteps) % 12

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, p_batch in tqdm(
            train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}"
        ):
            optimizer.zero_grad()
            weights = model(X_batch)
            y_pred = (weights * p_batch).sum(dim=1)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- eval: collect all test predictions, then compute metrics ---
        model.eval()
        all_preds, all_obs, all_baseline = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, p_batch in test_loader:
                weights = model(X_batch)
                y_pred = (weights * p_batch).sum(dim=1)
                baseline = p_batch.mean(dim=1)
                all_preds.append(y_pred.numpy())
                all_obs.append(y_batch.numpy())
                all_baseline.append(baseline.numpy())

        pred_grid = np.concatenate(all_preds).reshape(grid_shape)
        obs_grid = np.concatenate(all_obs).reshape(grid_shape)
        bl_grid = np.concatenate(all_baseline).reshape(grid_shape)

        fc_mae, fc_rmse, fc_acc, fc_nile = compute_metrics(
            pred_grid, obs_grid, test_months
        )
        bl_mae, bl_rmse, bl_acc, bl_nile = compute_metrics(
            bl_grid, obs_grid, test_months
        )
        test_loss = np.mean((pred_grid - obs_grid) ** 2)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}  train_loss={train_loss / len(train_loader):.4f}"
        )
        print(f"  {'':12} {'Forecast':>10} {'Baseline':>10}")
        print(f"  {'MAE':12} {fc_mae:10.4f} {bl_mae:10.4f}")
        print(f"  {'RMSE':12} {fc_rmse:10.4f} {bl_rmse:10.4f}")
        print(f"  {'ACC':12} {fc_acc:10.4f} {bl_acc:10.4f}")
        print(f"  {f'{N_QUANTILES}-ile Acc':12} {fc_nile:10.4f} {bl_nile:10.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "models/senate.pt")
            print("  -> saved best state to models/senate.pt")


if __name__ == "__main__":
    train()
