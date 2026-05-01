"""
Train a model from prepared ML data
"""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from util import AOI_ONLY


class PrecipDataset(Dataset):
    def __init__(self, path):
        print("loading data...")
        data = np.load(path)
        self.X = torch.tensor(data["X_input"], dtype=torch.float32)
        self.y = torch.tensor(data["y_obs"], dtype=torch.float32)
        self.p = torch.tensor(data["preds"], dtype=torch.float32)
        self.n_y = int(data["n_y"])
        self.n_x = int(data["n_x"])
        print(f"loaded {len(self.y):,} samples")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.p[idx]


class Senate(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.weight_head = nn.Linear(16, 7)
        self.resid_head = nn.Linear(16, 1)

    def forward(self, x):
        h = self.shared(x)
        weights = self.weight_head(h)
        residual = self.resid_head(h).squeeze(-1)
        return weights, residual


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
    data_path = "data/ml_data_aoi.npz" if AOI_ONLY else "data/ml_data.npz"
    dataset = PrecipDataset(data_path)

    print("splitting dataset...")
    train_timesteps = 300
    test_timesteps = 57
    n_spatial = dataset.n_y * dataset.n_x
    train_size = train_timesteps * n_spatial
    test_size = test_timesteps * n_spatial
    train_set = Subset(dataset, range(train_size))
    test_set = Subset(dataset, range(train_size, train_size + test_size))
    print(f"train: {train_size:,}  test: {test_size:,}")

    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)

    return (
        train_loader,
        test_loader,
        train_timesteps,
        test_timesteps,
        dataset.n_y,
        dataset.n_x,
    )


def eval_loss(mae, acc, nile_acc, alpha=0.4, beta=0.4, gamma=0.2):
    return alpha * mae - beta * acc - gamma * nile_acc


def train():
    model = Senate()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    EPOCHS = 100 if AOI_ONLY else 20
    print(f"\nstarting training for {EPOCHS} epochs...")
    best_eval = np.inf

    train_loader, test_loader, train_timesteps, test_timesteps, n_y, n_x = loaders()
    grid_shape = (test_timesteps, n_y, n_x)
    test_months = np.arange(train_timesteps, train_timesteps + test_timesteps) % 12

    model_path = "models/new_senate_aoi.pt" if AOI_ONLY else "models/new_senate.pt"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, p_batch in tqdm(
            train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}"
        ):
            optimizer.zero_grad()
            weights, residual = model(X_batch)
            y_pred = (weights * p_batch).sum(dim=1) + residual
            y_pred = F.relu(y_pred)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_obs, all_baseline = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, p_batch in test_loader:
                weights, residual = model(X_batch)
                y_pred = (weights * p_batch).sum(dim=1) + residual
                y_pred = F.relu(y_pred)
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

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}  train_loss={train_loss / len(train_loader):.4f}"
        )
        print(f"  {'':12} {'Forecast':>10} {'Baseline':>10}")
        print(f"  {'MAE':12} {fc_mae:10.4f} {bl_mae:10.4f}")
        print(f"  {'RMSE':12} {fc_rmse:10.4f} {bl_rmse:10.4f}")
        print(f"  {'ACC':12} {fc_acc:10.4f} {bl_acc:10.4f}")
        print(f"  {f'{N_QUANTILES}-ile Acc':12} {fc_nile:10.4f} {bl_nile:10.4f}")

        # normalise inputs to hybrid score
        acc_n = fc_acc * bl_mae / bl_acc
        nile_n = fc_nile * bl_mae / bl_nile

        eval_score = eval_loss(fc_mae, acc_n, nile_n)

        print(f"    [hybrid eval score: {eval_score:10.4f}]")

        if eval_score < best_eval:
            best_eval = eval_score
            torch.save(model.state_dict(), model_path)
            print(f"  -> saved best state to {model_path}")

    print(f"    [best eval score: {best_eval:10.4f}]")


if __name__ == "__main__":
    train()

