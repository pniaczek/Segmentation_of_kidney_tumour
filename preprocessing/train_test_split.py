import os
import random
import torch
import nibabel as nib

# Ścieżki
input_dir = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augumented_dataset"
output_dir = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augmented_data"
train_ratio = 0.8

# Zbierz unikalne ID przypadków, np. 'case_00585_cropped_image_aug00'
all_cases = sorted(list(set(
    fname.replace(".nii.gz", "").replace("_label", "_image")
    for fname in os.listdir(input_dir)
    if fname.endswith(".nii.gz") and "_image_" in fname
)))

# Podział na train/test
random.seed(42)
random.shuffle(all_cases)
split_idx = int(len(all_cases) * train_ratio)
train_cases = all_cases[:split_idx]
test_cases = all_cases[split_idx:]

# Funkcja do konwersji i zapisu jako .pt
def convert_and_save(cases, subset):
    subset_dir = os.path.join(output_dir, subset)
    os.makedirs(subset_dir, exist_ok=True)

    for case in cases:
        image_path = os.path.join(input_dir, f"{case}.nii.gz")
        label_case = case.replace("_image_", "_label_")
        label_path = os.path.join(input_dir, f"{label_case}.nii.gz")

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print(f"Pominięto {case} – brak pliku")
            continue

        # Wczytaj dane
        img = nib.load(image_path).get_fdata().astype("float32")
        seg = nib.load(label_path).get_fdata().astype("int64")  # label jako long (do one-hot)

        # Konwersja do tensora
        img_tensor = torch.tensor(img).unsqueeze(0)      # [1, D, H, W]
        label_tensor = torch.tensor(seg)                 # [D, H, W]

        # Zapis jako .pt
        output_path = os.path.join(subset_dir, f"{case}.pt")
        torch.save({"image": img_tensor, "label": label_tensor}, output_path)

# Wykonaj konwersję
convert_and_save(train_cases, "train")
convert_and_save(test_cases, "test")

print(f"Zapisano {len(train_cases)} przypadków w 'train/' i {len(test_cases)} w 'test/' w formacie .pt")
