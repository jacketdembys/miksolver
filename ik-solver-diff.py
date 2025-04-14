import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class DiffIKDataset(Dataset):
    def __init__(self, filename_D, filename_Q):
        # Load .npy files using NumPy
        self.q = torch.from_numpy(np.load(filename_Q)).float() #.to(device)
        self.pose = torch.from_numpy(np.load(filename_D)).float() #.to(device)
        
        print(f"Loaded Dataset:\nq.shape: {self.q.shape}\npose.shape: {self.pose.shape}")
        print(f"Loaded Dataset initially on --> device: {self.q.device}")

        assert self.q.shape[0] == self.pose.shape[0], "Mismatch in sample count"

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return {
            "q": self.q[idx],
            "pose": self.pose[idx],
        }
    


if __name__ == "__main__":

    
    # --- Configurable Parameters ---
    batch_size = 128
    max_epochs = 100
    disable_wandb = False  # Set True if using Weights & Biases
    dof = 7
    pose_dim = 7

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot = "panda"


    filename_Dtr = '/home/miksolver/ik_datasets/panda/endpoints_tr.npy'
    filename_Qtr = '/home/miksolver/ik_datasets/panda/samples_tr.npy'
    train_dataset = DiffIKDataset(filename_Dtr, filename_Qtr)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


    filename_Dte = '/home/miksolver/ik_datasets/panda/endpoints_te.npy'
    filename_Qte = '/home/miksolver/ik_datasets/panda/samples_te.npy'
    val_dataset = DiffIKDataset(filename_Dte, filename_Qte, device) 
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size)