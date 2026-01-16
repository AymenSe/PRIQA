import PIL
import torch
import pandas as pd
import numpy as np

class KADID10KDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=None, if_qf40=False, split='train', train_ratio=0.8, random_seed=42):
        """
        split: 'train', 'test', or None (all data)
        train_ratio: fraction of references to use for training
        """
        self.root_dir = root_dir
        self.transform = transform

        # Load CSV
        self.mos_data = pd.read_csv(csv_file)

        # Filter distorted images if needed
        if if_qf40:
            self.mos_data = self.mos_data[
                self.mos_data['dist_img'].str.contains(r'_10_(?:01|02|03)', regex=True)
            ]
        else:
            self.mos_data = self.mos_data[self.mos_data['dist_img'].str.contains('_10_')]

        self.mos_data = self.mos_data.reset_index(drop=True)

        # --- Split by reference image ---
        if split in ['train', 'test']:
            # Build dict of reference -> list of row indices
            ref_dict = {}
            for idx, ref_name in enumerate(self.mos_data['ref_img']):
                if ref_name not in ref_dict:
                    ref_dict[ref_name] = []
                ref_dict[ref_name].append(idx)

            ref_keys = list(ref_dict.keys())
            np.random.seed(random_seed)
            np.random.shuffle(ref_keys)
            n_train = int(len(ref_keys) * train_ratio)

            train_refs = ref_keys[:n_train]
            test_refs = ref_keys[n_train:]

            if split == 'train':
                selected_refs = train_refs
            else:
                selected_refs = test_refs

            # Keep only rows corresponding to selected references
            selected_indices = [i for r in selected_refs for i in ref_dict[r]]
            self.mos_data = self.mos_data.iloc[selected_indices].reset_index(drop=True)

            print(f"[INFO] {split.upper()} dataset: {len(self.mos_data)} samples from {len(selected_refs)} reference images.")

    def __len__(self):
        return len(self.mos_data)

    def __getitem__(self, idx):
        row = self.mos_data.iloc[idx]
        dist_path, ref_path, quality_score = row['dist_img'], row['ref_img'], row['dmos']

        distorted_image = PIL.Image.open(f"{self.root_dir}/Dist/{dist_path}").convert("RGB")
        reference_image = PIL.Image.open(f"{self.root_dir}/Refs/{ref_path}").convert("RGB")

        if self.transform:
            distorted_image = self.transform(distorted_image)
            reference_image = self.transform(reference_image)

        return distorted_image, reference_image, quality_score, dist_path


# def build_model(device: torch.device) -> nn.Module:
#     model = SwinIR(
#         upscale=1,
#         in_chans=3,
#         img_size=126,
#         window_size=7,
#         img_range=255.,
#         depths=[6, 6, 6, 6, 6, 6],
#         embed_dim=180,
#         num_heads=[6, 6, 6, 6, 6, 6],
#         mlp_ratio=2,
#         upsampler='',
#         resi_connection='1conv'
#     )
#     print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#     return model.to(device)

# if __name__ == "__main__":
#     from torchvision import transforms

#     transform = transforms.Compose([transforms.ToTensor()])

#     dataset = KADID10KDataset(
#         root_dir="kadid_jpeg",
#         csv_file="kadid_jpeg/dmos.csv",
#         transform=transform,
#         if_qf40=False,
#         split='train'
#     )

#     model = build_model(torch.device("cuda:2" if torch.cuda.is_available() else "cpu"))
#     model.eval()
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#     for i, (dist_img, ref_img, mos, dist_path) in enumerate(data_loader):
#         dist_img = dist_img.to(model.device)
#         with torch.no_grad():
#             pref_img = model(dist_img)
#         print(f"Processed {i+1}/{len(dataset)}: {dist_path[0]}")