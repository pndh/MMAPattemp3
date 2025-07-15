import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.UNI import UNI
from dataset import UNI_HER2ST
from sklearn.cluster import KMeans
import os
from tqdm import tqdm
import statistics

# ---------------------------
# Feature Extraction Dataset
# ---------------------------

class UNI_FeatureDataset(Dataset):
    def __init__(self, dataset, name):
        """
        dataset: instance của UNI_HER2ST đã load sẵn
        name: tên WSI (VD: 'A2')
        """
        self.dataset = dataset
        self.name = name

        self.img = self.dataset.get_img(name)
        self.centers = self.dataset.center_dict[name]
        self.r = self.dataset.r
        self.transform = self.dataset.transform  

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        x, y = self.centers[idx]
        patch = self.img.crop((x - self.r, y - self.r, x + self.r, y + self.r))
        patch = np.array(patch)

        ms_patches = self.dataset.scale_crop(patch)

        patch_0 = self.transform(ms_patches[0])
        patch_1 = self.transform(ms_patches[1])
        patch_2 = self.transform(ms_patches[2])

        return patch_0, patch_1, patch_2


# ---------------------------
# Main Processing Pipeline
# ---------------------------

def extract_features():
    # Hyperparameters
    fold = 5
    batch_size = 1
    num_workers = 4
    device = torch.device("cuda")
    cache_dir = 'cache_features_test/'

    os.makedirs(cache_dir, exist_ok=True)

    # Load dataset + model
    dataset = UNI_HER2ST(train=False, fold=fold)
    model = UNI.load_from_checkpoint(
        "model_ckpts/UNI_final/UNI_every5epoch_-htg_her2st_785_32_cv_5_epoch=44.ckpt",
        n_genes=785, learning_rate=1e-5, max_epochs=50
    )
    model.eval().to(device)

    # Statistics tracker
    all_lengths = []

    for name in dataset.names:
        print(f"\nProcessing WSI: {name}")

        feat_dataset = UNI_FeatureDataset(dataset, name)
        loader = DataLoader(feat_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        outs, clss = [], []

        with torch.no_grad():
            for b0, b1, b2 in tqdm(loader, desc=f"Extracting features for {name}"):
                b0, b1, b2 = b0.to(device), b1.to(device), b2.to(device)
                out, cls, _ = model(b0, b1, b2)
                outs.append(out.cpu())
                clss.append(cls.cpu())

        outs = torch.cat(outs).numpy()
        clss = torch.cat(clss).numpy()

        # Save outputs
        np.save(f"{cache_dir}/{name}_outs.npy", outs)
        np.save(f"{cache_dir}/{name}_clss.npy", clss)

        # Clustering
        n_clusters = min(max(len(feat_dataset) // 5, 32), 80)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(clss)
        np.save(f"{cache_dir}/{name}_cluster_centers.npy", kmeans.cluster_centers_)

        # Các tọa độ patch ban đầu
        centers_np = np.array(dataset.center_dict[name])  # shape: [num_patches, 2]
        assert clss.shape[0] == centers_np.shape[0], "Mismatch between features and coordinates"

        # ----------------------
        # 1. Vị trí trung bình (mean)
        # ----------------------
        cluster_labels = kmeans.labels_                            # [num_patches]
        cluster_centers_loc_mean = np.zeros((n_clusters, 2))       # [n_clusters, 2]

        for i in range(n_clusters):
            cluster_points = centers_np[cluster_labels == i]
            if len(cluster_points) > 0:
                cluster_centers_loc_mean[i] = cluster_points.mean(axis=0)
            else:
                cluster_centers_loc_mean[i] = np.array([0, 0])  # fallback (shouldn't happen)

        np.save(f"{cache_dir}/{name}_cluster_centers_loc_mean.npy", cluster_centers_loc_mean)

        # ----------------------
        # 2. Vị trí gần nhất với cluster center (nearest)
        # ----------------------
        from scipy.spatial.distance import cdist

        distances = cdist(kmeans.cluster_centers_, clss)           # [n_clusters, n_patches]
        closest_idx = distances.argmin(axis=1)                     # [n_clusters]
        cluster_centers_loc_nearest = centers_np[closest_idx]      # [n_clusters, 2]

        np.save(f"{cache_dir}/{name}_cluster_centers_loc_nearest.npy", cluster_centers_loc_nearest)

        all_lengths.append(len(feat_dataset))

    # Print final statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total WSI processed: {len(all_lengths)}")
    print(f"Min centers per WSI: {min(all_lengths)}")
    print(f"Max centers per WSI: {max(all_lengths)}")
    print(f"Average centers per WSI: {statistics.mean(all_lengths):.2f}")


# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    extract_features()
