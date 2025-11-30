import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def build_manifest(data_dir, output_path):
    """
    Scans the data directory for PD and NPD folders and creates a manifest CSV.
    Expected structure:
    data_dir/
      PD/
        subject001/
          img_0001.png
      NPD/
        subject101/
          ...
    """
    records = []
    
    # Check if root directories exist
    pd_dir = os.path.join(data_dir, 'PD')
    npd_dir = os.path.join(data_dir, 'NPD')
    
    if not os.path.exists(pd_dir) and not os.path.exists(npd_dir):
        print(f"Warning: PD/NPD directories not found in {data_dir}. Checking for flat structure or other variants.")
        # Create empty manifest
        df = pd.DataFrame(columns=['image_path', 'subject_alias', 'label'])
        df.to_csv(output_path, index=False)
        return df

    for label_str, label_val in [('PD', 1), ('NPD', 0)]:
        class_dir = os.path.join(data_dir, label_str)
        if not os.path.exists(class_dir):
            print(f"Class dir {class_dir} not found")
            continue
            
        # List subjects
        subjects = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
        print(f"Found {len(subjects)} subjects in {label_str}")
        
        for subj in subjects:
            subj_dir = os.path.join(class_dir, subj)
            
            # Recursive search for images
            images = []
            for root, _, files in os.walk(subj_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        images.append(os.path.join(root, f))
            
            if not images:
                # print(f"Warning: No images found for subject {subj} in {label_str}")
                continue
                
            for img_path in images:
                records.append({
                    'image_path': img_path,
                    'subject_alias': subj,
                    'label': label_val
                })
                
    df = pd.DataFrame(records)
    if len(df) == 0:
        df = pd.DataFrame(columns=['image_path', 'subject_alias', 'label'])
        
    df.to_csv(output_path, index=False)
    print(f"Manifest saved to {output_path}. Total images: {len(df)}")
    return df

class ParkinsonDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        self.manifest = pd.read_csv(manifest_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img_path = row['image_path']
        label = row['label']
        subject = row['subject_alias']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor or handle error? 
            # For simplicity, let's return a zero tensor or raise
            raise e
            
        if self.transform:
            image = self.transform(image)
            
        return image, label, subject, img_path

def get_transforms():
    """
    Returns the standard ImageNet preprocessing transforms.
    Resize to 224x224, Normalize.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    # Test run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ntua-parkinson-dataset-master')
    parser.add_argument('--output', type=str, default='outputs/manifest.csv')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build_manifest(args.data_dir, args.output)
