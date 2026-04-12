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
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 7),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def loaders():
    dataset = PrecipDataset("data/ml_data.npz")

    print("splitting dataset...")
    train_timesteps = 300
    test_timesteps = 60
    train_size = train_timesteps * 179 * 360
    test_size = test_timesteps * 179 * 360
    train_set = Subset(dataset, range(train_size))
    test_set = Subset(dataset, range(train_size, train_size + test_size))
    print(f"train: {train_size:,}  test: {test_size:,}")

    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)

    return train_loader, test_loader


def train():
    model = Senate()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    EPOCHS = 20
    print(f"\nstarting training for {EPOCHS} epochs...")
    best_loss = np.inf

    train_loader, test_loader = loaders()

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
        model.eval()
        test_loss = 0.0
        baseline_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, p_batch in test_loader:
                weights = model(X_batch)
                y_pred = (weights * p_batch).sum(dim=1)
                test_loss += loss_fn(y_pred, y_batch).item()
                baseline = p_batch.mean(dim=1)
                baseline_loss += loss_fn(baseline, y_batch).item()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}  train={train_loss / len(train_loader):.4f}  test={test_loss / len(test_loader):.4f}  baseline={baseline_loss / len(test_loader):.4f}"
        )
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "models/senate.pt")
            print("saved best state to models/senate.pt")


if __name__ == "__main__":
    train()
