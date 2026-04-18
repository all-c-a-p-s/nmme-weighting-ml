"""
Load model and save predictions to a file + functions to run model on new data
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from train import Senate, PrecipDataset
import xarray as xr

model = Senate()
model.load_state_dict(torch.load("models/senate.pt"))
model.eval()


def encode(month, y, x, ginis):
    coords = [
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * x / 360),
        np.cos(2 * np.pi * x / 360),
        np.sin(np.pi * (y + 90) / 180),
        np.cos(np.pi * (y + 90) / 180),
    ]
    return torch.tensor(coords + list(ginis), dtype=torch.float32)


def get_weights(month, y, x, ginis):
    x_input = encode(month, y, x, ginis).unsqueeze(0)
    with torch.no_grad():
        weights = model(x_input).squeeze(0).numpy()
    return weights


def forecast(month, y, x, nmme_preds, ginis):
    weights = get_weights(month, y, x, ginis)
    return np.dot(weights, nmme_preds)


dataset = PrecipDataset("data/ml_data.npz")
train_size = 300 * 179 * 360
test_size = 60 * 179 * 360
test_set = Subset(dataset, range(train_size, train_size + test_size))
test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)

print("generating test period predictions...")
all_preds = []
all_obs = []
all_baseline = []

with torch.no_grad():
    for X_batch, y_batch, p_batch in test_loader:
        weights = model(X_batch)
        y_pred = (weights * p_batch).sum(dim=1)
        all_preds.append(y_pred.numpy())
        all_obs.append(y_batch.numpy())
        all_baseline.append(p_batch.mean(dim=1).numpy())

all_preds = np.concatenate(all_preds).reshape(60, 179, 360)
all_obs = np.concatenate(all_obs).reshape(60, 179, 360)
all_baseline = np.concatenate(all_baseline).reshape(60, 179, 360)

Ys = np.arange(-89, 90, 1.0)
Xs = np.arange(0, 360, 1.0)

preds_da = xr.DataArray(all_preds, dims=["time", "Y", "X"], coords={"Y": Ys, "X": Xs})
obs_da = xr.DataArray(all_obs, dims=["time", "Y", "X"], coords={"Y": Ys, "X": Xs})
baseline_da = xr.DataArray(
    all_baseline, dims=["time", "Y", "X"], coords={"Y": Ys, "X": Xs}
)

ds = xr.Dataset({"forecast": preds_da, "obs": obs_da, "baseline": baseline_da})
ds.to_netcdf("data/test_predictions.nc")
print("saved to data/test_predictions.nc")
