import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class TID2019Dataset(Dataset):
    def __init__(self, root_dir, train=True, train_ratio=0.8, seed=42, transform=None):
        """
        Args:
            root_dir (str): Path to the root 'tid2019' directory.
            train (bool): If True, returns the training set. If False, returns the test set.
            train_ratio (float): Percentage of reference images to use for training (0.0 to 1.0).
            seed (int): Seed for reproducibility to ensure train/test splits don't overlap.
            transform (callable, optional): Transform to be applied on the images.
        """
        self.root_dir = root_dir
        self.distorted_dir = os.path.join(root_dir, 'distorted_images')
        self.reference_dir = os.path.join(root_dir, 'reference_images')
        self.transform = transform
        self.data = []

        # 1. Load all lines from the text file
        txt_path = os.path.join(root_dir, 'mos_with_names.txt')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Could not find label file at {txt_path}")

        all_entries = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    mos_score = float(parts[0])
                    img_name = " ".join(parts[1:])
                    
                    # FILTER: Only keep images with _10_ (JPEG distortion)
                    if "_10_" in img_name:
                        # Parse Reference ID (e.g., 'i01' from 'i01_10_1.bmp')
                        # We need this ID to group images for the split
                        ref_id = img_name.split('_')[0].lower() # normalizing to lower case for grouping
                        all_entries.append({
                            'name': img_name, 
                            'mos': mos_score, 
                            'ref_id': ref_id
                        })

        # 2. Identify Unique Reference Images
        # We find all unique IDs (e.g., i01, i02, ... i25)
        unique_ref_ids = sorted(list(set(entry['ref_id'] for entry in all_entries)))
        
        # 3. Shuffle and Split Reference IDs
        # We shuffle the IDs themselves, not the images, to prevent leakage.
        random.seed(seed)
        random.shuffle(unique_ref_ids)
        
        split_idx = int(len(unique_ref_ids) * train_ratio)
        
        if train:
            selected_ref_ids = set(unique_ref_ids[:split_idx])
        else:
            selected_ref_ids = set(unique_ref_ids[split_idx:])

        # 4. Filter the main data list based on selected IDs
        # Only add images whose reference ID is in our selected set (train or test)
        for entry in all_entries:
            if entry['ref_id'] in selected_ref_ids:
                self.data.append((entry['name'], entry['mos']))

        # Debug print to verify split
        mode_name = "Train" if train else "Test"
        print(f"[{mode_name} Split] Loaded {len(self.data)} images from {len(selected_ref_ids)} unique reference scenes.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dist_img_name, mos = self.data[idx]
        
        # Define paths
        dist_img_path = os.path.join(self.distorted_dir, dist_img_name)
        
        # Derive Reference Image Path
        filename_parts = dist_img_name.split('_')
        ref_id = filename_parts[0] 
        ref_number = ref_id[1:] 
        
        # Try standard uppercase reference name "I01.BMP"
        ref_img_name = f"I{ref_number}.BMP"
        ref_img_path = os.path.join(self.reference_dir, ref_img_name)
        
        # Fallback for Linux case-sensitivity
        if not os.path.exists(ref_img_path):
            ref_img_path_lower = os.path.join(self.reference_dir, f"i{ref_number}.bmp")
            if os.path.exists(ref_img_path_lower):
                ref_img_path = ref_img_path_lower
            else:
                ref_img_path = os.path.join(self.reference_dir, f"{ref_id}.bmp")

        # Load
        try:
            dist_image = Image.open(dist_img_path).convert('RGB')
            ref_image = Image.open(ref_img_path).convert('RGB')
        except FileNotFoundError as e:
            print(f"Error loading:\nDist: {dist_img_path}\nRef: {ref_img_path}")
            raise e

        # Transform
        if self.transform:
            dist_image = self.transform(dist_image)
            ref_image = self.transform(ref_image)

        mos = torch.tensor(mos, dtype=torch.float32)

        return dist_image, ref_image, mos, dist_img_name



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create Training Set (80% of reference scenes)
    train_dataset = TID2019Dataset(
        root_dir='./tid2019', 
        train=True,           # <--- Request Train Split
        train_ratio=0.8,      # 80% split
        seed=42,              # Fixed seed ensures no overlap with test
        transform=transform
    )

    # Create Test Set (The remaining 20% of reference scenes)
    test_dataset = TID2019Dataset(
        root_dir='./tid2019', 
        train=False,          # <--- Request Test Split
        train_ratio=0.8,      # Must match train split ratio
        seed=42,              # Must match train seed
        transform=transform
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size:  {len(test_dataset)}")