import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import pandas as pd

# === ŚCIEŻKI ===
input_root = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/raw"
output_dir = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/visualizations"
os.makedirs(output_dir, exist_ok=True)

# === DEFINICJA KLAS ===
class_labels = [0, 1, 2, 3]
class_names = ["background", "kidney", "tumor", "cyst"]
voxel_summary = []

# === ANALIZA MASEK ===
for case in tqdm(sorted(os.listdir(input_root))):
    mask_path = os.path.join(input_root, case, "instances", "segmentation.nii.gz")
    if not os.path.exists(mask_path):
        continue
    mask = nib.load(mask_path).get_fdata().astype(np.int32)

    row = {"case": case}
    for cls, name in zip(class_labels, class_names):
        voxel_count = int((mask == cls).sum())
        row[name] = voxel_count
    voxel_summary.append(row)

# === KONWERSJA DO DATAFRAME i CSV ===
df = pd.DataFrame(voxel_summary)
csv_path = os.path.join(output_dir, "voxel_counts_summary.csv")
df.to_csv(csv_path, index=False)
print(f"[✓] Zapisano CSV: {csv_path}")

# === WYKRESY HISTOGRAMÓW ===
for name in class_names:
    counts = df[name].values
    if all(c == 0 for c in counts):
        print(f"[!] Klasa {name} nie występuje w żadnym przypadku.")
        continue

    plt.figure()
    plt.hist(counts, bins=30, color="skyblue", edgecolor="black")
    plt.title(f"Rozkład objętości klasy: {name}")
    plt.xlabel("Liczba voxelów")
    plt.ylabel("Liczba przypadków")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"hist_class_{name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[✓] Zapisano histogram: {out_path}")
