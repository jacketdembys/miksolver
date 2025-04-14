import torch
import torch.nn as nn
import numpy as np
import os
import wandb
from torch.utils.data import Dataset, DataLoader
from utils import get_robot_choice, reconstruct_pose_modified

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

# --- Model Architecture ---
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

# --- Inference Execution ---
def run_inference(artifact_path, data_dir, robot_name="panda", batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Running inference on device: {device}]")

    # Load test set
    dataset = DiffIKDataset(
        os.path.join(data_dir, "endpoints_te.npy"),
        os.path.join(data_dir, "samples_te.npy")
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    # Load model
    model = DiffIKDenoiser()
    model.load_state_dict(torch.load(artifact_path, map_location=device))
    model.to(device)

    y_preds, y_desireds = [], []
    for batch in loader:
        pose = batch["pose"].to(device)
        q_pred = sample(model, pose)
        y_preds.append(q_pred.detach().cpu().numpy())
        y_desireds.append(batch["q"].numpy())

    y_preds = np.concatenate(y_preds, axis=0)
    y_desireds = np.concatenate(y_desireds, axis=0)

    # Reconstruct and evaluate
    robot_choice = get_robot_choice(robot_name)
    X_desireds, X_preds, X_errors = reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
    X_errors_report = np.array([[X_errors.min(axis=0)],
                                [X_errors.mean(axis=0)],
                                [X_errors.max(axis=0)],
                                [X_errors.std(axis=0)]]).squeeze()

    print("\n--- Inference Results ---")
    print("Mean Position Error (mm):", (X_errors_report[1, :3] * 1000).mean())
    print("Mean Orientation Error (deg):", np.rad2deg(X_errors_report[1, 3:]).mean())

    return y_preds, X_preds, X_errors_report

if __name__ == "__main__":
    # Replace with the path to your wandb-downloaded artifact and test dataset directory
    artifact_model_path = "results/best_epoch_XXX_pose_YY.YY.pth"
    test_data_dir = "/home/miksolver/ik_datasets/panda"
    run_inference(artifact_model_path, test_data_dir)
