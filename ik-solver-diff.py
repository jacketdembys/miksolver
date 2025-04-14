import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import sys
from utils import *

# --- Dataset Loader ---
class DiffIKDataset(Dataset):
    def __init__(self, filename_D, filename_Q):
        self.q = torch.from_numpy(np.load(filename_Q)).float()
        self.pose = torch.from_numpy(np.load(filename_D)).float()
        assert self.q.shape[0] == self.pose.shape[0]

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return {"q": self.q[idx], "pose": self.pose[idx]}

# --- Diffusion MLP Architecture ---
class DiffIKDenoiser(nn.Module):
    def __init__(self, dof=7, pose_dim=7, hidden_dim=512, time_embed_dim=64):
        super().__init__()
        self.dof = dof
        self.num_timesteps = 1000
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        input_dim = dof + pose_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, dof)
        )

    def forward(self, q_t, pose, t):
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        x = torch.cat([q_t, pose, t_embed], dim=-1)
        return self.net(x)

# --- Sampling Function ---
@torch.no_grad()
def sample(model, pose, ddim_steps=50):
    model.eval()
    B = pose.shape[0]
    q = torch.randn(B, model.dof).to(pose.device)
    pose = pose.repeat_interleave(1, dim=0)
    for t in reversed(range(1, ddim_steps + 1)):
        t_tensor = torch.full((q.size(0),), t, device=q.device, dtype=torch.long)
        beta = 1e-4 + (t_tensor / model.num_timesteps).unsqueeze(-1)
        noise_pred = model(q, pose, t_tensor)
        q = q - beta * noise_pred
    return q

# --- Validation ---
def validate(model, val_loader, device, robot_name):
    model.eval()
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    robot_choice = get_robot_choice(robot_name)
    with torch.no_grad():
        for batch in val_loader:
            q_gt = batch["q"].to(device)
            pose_gt = batch["pose"].to(device)

            # compute denoising loss
            noise = torch.randn_like(q_gt)
            t = torch.randint(0, model.num_timesteps, (q_gt.size(0),), device=device)
            beta = 1e-4 + (t / model.num_timesteps).unsqueeze(-1)
            q_t = q_gt + beta * noise
            noise_pred = model(q_t, pose_gt, t)
            loss = loss_fn(noise_pred, noise)
            total_loss += loss.item()
            monitored_total_loss = total_loss / len(val_loader)

            # compute prediction and FK error
            q_pred = sample(model, pose_gt)
            y_preds = q_pred.detach().cpu().numpy()
            y_desireds = q_gt.detach().cpu().numpy()

            print(q_pred.shape)
            X_desireds, X_preds, X_errors = reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
            X_errors_report = np.array([[X_errors.min(axis=0)],
                                        [X_errors.mean(axis=0)],
                                        [X_errors.max(axis=0)],
                                        [X_errors.std(axis=0)]]).squeeze()
            
            
            X_errors_r = X_errors[:,:6]
            X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
            X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:]) 
            avg_position_error = X_errors_r[1,:3].mean()
            avg_orientation_error = X_errors_r[1,3:].mean()

            print("avg_position_error (mm): {}".format(avg_position_error))
            print("avg_orientation_error (deg): {}".format(avg_orientation_error))

            
            
            results = {
                "y_preds": y_preds,
                "X_preds": X_preds,
                "y_desireds": y_desireds,
                "X_desireds": X_desireds,
                "X_errors": X_errors,
                "X_errors_report": X_errors_report
            }
            sys.exit()

    return monitored_total_loss

# --- Training Loop ---
def train_loop(model, train_loader, val_loader, max_epochs=10, lr=1e-4, robot_name="panda"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training on device: {device}]")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{max_epochs}]")
        for batch in train_loader:
            q = batch["q"].to(device)
            pose = batch["pose"].to(device)
            noise = torch.randn_like(q)
            t = torch.randint(0, model.num_timesteps, (q.size(0),), device=device)
            beta = 1e-4 + (t / model.num_timesteps).unsqueeze(-1)
            q_t = q + beta * noise
            noise_pred = model(q_t, pose, t)
            loss = loss_fn(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, device, robot_name)
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# --- Main ---
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 128
    max_epochs = 1
    dof = 7
    pose_dim = 7
    robot_name = "panda"

    dataset_path = f"/home/miksolver/ik_datasets/{robot_name}"
    train_dataset = DiffIKDataset(
        os.path.join(dataset_path, "endpoints_tr.npy"),
        os.path.join(dataset_path, "samples_tr.npy")
    )
    val_dataset = DiffIKDataset(
        os.path.join(dataset_path, "endpoints_te.npy"),
        os.path.join(dataset_path, "samples_te.npy")
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DiffIKDenoiser(dof=dof, pose_dim=pose_dim)
    train_loop(model, train_loader, val_loader, max_epochs=max_epochs, robot_name=robot_name)
