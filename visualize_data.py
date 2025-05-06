import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Load CSV ---
file_path = "../for_docker/left-out-datasets/7DoF-Combined/review_data_7DoF-7R-PA10_1000000_qlim_scale_10_seq_1.csv"  # <-- Change this to your actual CSV file path
df = pd.read_csv(file_path)

# --- Convert to NumPy ---
data = df.to_numpy(dtype=np.float32)

# --- Split ---
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# --- Extract Positions ---
train_xyz = train_data[:, 0:3]  # x_p, y_p, z_p
val_xyz = val_data[:, 0:3]

# --- Plot ---
fig = plt.figure(figsize=(12, 6))

# Train subplot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(train_xyz[:, 0], train_xyz[:, 1], train_xyz[:, 2], c='blue', alpha=0.6)
ax1.set_title("Train Positions (X, Y, Z)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Validation subplot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(val_xyz[:, 0], val_xyz[:, 1], val_xyz[:, 2], c='red', alpha=0.6)
ax2.set_title("Validation Positions (X, Y, Z)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.tight_layout()
plt.show()
