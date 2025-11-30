import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import ParkinsonDataset, get_transforms, build_manifest

def get_backbone(name='resnet18'):
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        embedding_dim = 512
        # Remove fc layer
        model.fc = nn.Identity()
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        embedding_dim = 1280
        model.classifier = nn.Identity()
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        embedding_dim = 1280
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return model, embedding_dim

def extract_features(data_dir, output_dir, backbone_name='resnet18', batch_size=8, num_workers=2, fine_tune=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure manifest exists
    manifest_path = os.path.join(output_dir, 'manifest.csv')
    if not os.path.exists(manifest_path):
        print("Manifest not found, creating...")
        build_manifest(data_dir, manifest_path)
        
    dataset = ParkinsonDataset(manifest_path, transform=get_transforms())
    if len(dataset) == 0:
        print("No images found in dataset. Exiting extraction.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model, dim = get_backbone(backbone_name)
    model.to(device)
    
    if not fine_tune:
        model.eval()
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
            
    embeddings = []
    image_paths = []
    
    print(f"Starting feature extraction with {backbone_name}...")
    
    with torch.no_grad():
        for images, labels, subjects, paths in tqdm(loader):
            images = images.to(device)
            # Forward pass
            emb = model(images)
            # Flatten if necessary (some models might return spatial maps if not handled)
            # ResNet18 with fc=Identity returns (B, 512)
            # MobileNet/EfficientNet with classifier=Identity returns (B, 1280)
            
            emb = emb.view(emb.size(0), -1)
            embeddings.append(emb.cpu().numpy())
            image_paths.extend(paths)
            
    all_embeddings = np.vstack(embeddings).astype(np.float32)
    
    # Save
    emb_path = os.path.join(output_dir, 'embeddings.npy')
    np.save(emb_path, all_embeddings)
    
    print(f"Saved embeddings to {emb_path}")
    print(f"Shape: {all_embeddings.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ntua-parkinson-dataset-master')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v2', 'efficientnet_b0'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--fine_tune', action='store_true')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    extract_features(args.data_dir, args.output_dir, args.backbone, args.batch_size, args.num_workers, args.fine_tune)
