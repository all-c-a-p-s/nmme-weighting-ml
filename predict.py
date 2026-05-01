"""
Load model and save predictions to a file + functions to run model on new data
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from train import Senate, PrecipDataset
import xarray as xr
from util import AOI_ONLY

data_path = "data/ml_data_aoi.npz" if AOI_ONLY else "data/ml_data.npz"
model_path = "models/senate_aoi.pt" if AOI_ONLY else "models/senate.pt"
output_path = "data/test_predictions_aoi.nc" if AOI_ONLY else "data/test_predictions.nc"

model = Senate()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


# this should be the month you are predicting FOR
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


def get_weights_and_residual(month, y, x, ginis):
    x_input = encode(month, y, x, ginis).unsqueeze(0)

    with torch.no_grad():
        weights, residual = model(x_input)

    return weights.squeeze(0).numpy(), residual.squeeze().item()


def forecast(month, y, x, nmme_preds, ginis):
    weights, residual = get_weights_and_residual(month, y, x, ginis)
    raw_pred = np.dot(weights, nmme_preds) + residual
    return max(0.0, raw_pred)


dataset = PrecipDataset(data_path)
n_y, n_x = dataset.n_y, dataset.n_x

train_size = 300 * n_y * n_x
test_size = 57 * n_y * n_x

test_set = Subset(dataset, range(train_size, train_size + test_size))
test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)

print("generating test period predictions...")

all_preds = []
all_obs = []
all_baseline = []

with torch.no_grad():
    for X_batch, y_batch, p_batch in test_loader:
        weights, residual = model(X_batch)

        residual = residual.squeeze(-1)
        raw_pred = (weights * p_batch).sum(dim=1) + residual
        y_pred = torch.relu(raw_pred)

        all_preds.append(y_pred.numpy())
        all_obs.append(y_batch.numpy())
        all_baseline.append(p_batch.mean(dim=1).numpy())

raw = np.load(data_path)
Ys = raw["Ys"]
Xs = raw["Xs"]

all_preds = np.concatenate(all_preds).reshape(57, n_y, n_x)
all_obs = np.concatenate(all_obs).reshape(57, n_y, n_x)
all_baseline = np.concatenate(all_baseline).reshape(57, n_y, n_x)

preds_da = xr.DataArray(
    all_preds,
    dims=["time", "Y", "X"],
    coords={"Y": Ys, "X": Xs},
)

obs_da = xr.DataArray(
    all_obs,
    dims=["time", "Y", "X"],
    coords={"Y": Ys, "X": Xs},
)

baseline_da = xr.DataArray(
    all_baseline,
    dims=["time", "Y", "X"],
    coords={"Y": Ys, "X": Xs},
)

ds = xr.Dataset(
    {
        "forecast": preds_da,
        "obs": obs_da,
        "baseline": baseline_da,
    }
)

ds.to_netcdf(output_path)
print(f"saved to {output_path}")
