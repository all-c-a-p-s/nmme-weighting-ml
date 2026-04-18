"""
Just a script I'm rewriting to look into the data as needed
"""

import torch
import numpy as np
from train import Senate

model = Senate()
model.load_state_dict(torch.load("models/senate.pt"))
model.eval()

models = [
    "canesm5",
    "cola-ccsm4",
    "cola-cesm1",
    "gem5p2",
    "gfdl-spear",
    "nasa-geos",
    "ncep-cfsv2",
]


def get_test_weights(month, lat, lon, ginis):
    m_val = month % 12
    coords = [
        np.sin(2 * np.pi * m_val / 12),
        np.cos(2 * np.pi * m_val / 12),
        np.sin(2 * np.pi * lon / 360),
        np.cos(2 * np.pi * lon / 360),
        np.sin(np.pi * (lat + 90) / 180),
        np.cos(np.pi * (lat + 90) / 180),
    ]
    x_input = torch.tensor(coords + ginis, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        weights = model(x_input).squeeze(0).numpy()

    print(f"\n--- Scenario: Month {month}, Lat {lat}, Lon {lon} ---")
    for name, w, g in zip(models, weights, ginis):
        print(f"{name:16} | Weight: {w:.4f} | Gini (Uncertainty): {g:.2f}")


# CASE 1: Perfect Agreement (All Ginis = 0)
# Does the model favor a specific 'reliable' model by default?
get_test_weights(month=6, lat=0, lon=0, ginis=[0.0] * 7)

# CASE 2: High Uncertainty in Model 0 (CanESM5)
# Does the weight for CanESM5 drop compared to Case 1?
ginis_case2 = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
get_test_weights(month=6, lat=0, lon=0, ginis=ginis_case2)

# CASE 3: Only GFDL-Spear is "Certain" (Model 4)
# Does the weight shift heavily toward GFDL?
ginis_case3 = [0.8, 0.8, 0.8, 0.8, 0.05, 0.8, 0.8]
get_test_weights(month=1, lat=45, lon=260, ginis=ginis_case3)

# CASE 4: The "Poles" (High Latitude)
# Check if the model has a different spatial preference for the same Ginis
get_test_weights(month=6, lat=80, lon=0, ginis=[0.2] * 7)
