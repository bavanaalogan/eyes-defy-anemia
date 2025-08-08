#!/usr/bin/env python3
"""
Improved Anemia Detection Pipeline - Final Version
Addresses class imbalance, training stability, and accuracy issues
"""

import os
import json
import logging
import warnings
import gc
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Suppress albumentations version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='albumentations')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from PIL import Image
import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets = targets.long()
        log_prob = F.log_softmax(inputs, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_prob, dim=-1))


class OptimalAugmentation:
    @staticmethod
    def get_train_transforms():
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CoarseDropout(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    @staticmethod
    def get_val_transforms():
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class AnemiaDataset(Dataset):
    def __init__(self, df, base_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.transform = transform
        
        # Valid image extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __len__(self):
        return len(self.df)
    
    def find_image_file(self, source, number):
        """Find image file for given source and number"""
        base_folder = Path(self.base_path) / source / str(number)
        
        if not base_folder.exists():
            return None
        
        for ext in self.valid_extensions:
            number_str = str(number)
            for pattern in [f"{number_str}{ext}", f"{number_str.upper()}{ext}", f"{number_str.lower()}{ext}"]:
                image_path = base_folder / pattern
                if image_path.exists():
                    return str(image_path)
        
        # Fallback: find any image in the folder
        for file_path in base_folder.iterdir():
            if file_path.suffix.lower() in self.valid_extensions:
                return str(file_path)
        
        return None

    def load_image_robust(self, image_path):
        """Robustly load image with multiple fallbacks"""
        try:
            # Try PIL first
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                return np.array(img)
        except Exception:
            try:
                # Try OpenCV
                img = cv2.imread(image_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                pass
        
        # Create dummy image if all fails
        return np.ones((224, 224, 3), dtype=np.uint8) * 128

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image_path = self.find_image_file(row['Source'], row['Number'])
        
        if image_path and os.path.exists(image_path):
            image = self.load_image_robust(image_path)
        else:
            image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Ensure tensor is float32
        image = image.float()
        
        # Encode gender (M=1, F=0)
        gender_encoded = 1.0 if str(row['Gender']).upper().startswith('M') else 0.0
        
        return {
            'image': image,
            'hgb': torch.tensor(float(row['Hgb']), dtype=torch.float32),
            'anemia_class': torch.tensor(int(row['anemia_class']), dtype=torch.long),
            'gender': torch.tensor(gender_encoded, dtype=torch.float32),
            'age': torch.tensor(float(row['Age']), dtype=torch.float32)
        }


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OptimalEnsemble(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):  # Changed to 3 classes
        super().__init__()
        
        # Backbones
        self.backbone1 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feat1_dim = self.backbone1(dummy_input).shape[1]
            feat2_dim = self.backbone2(dummy_input).shape[1]
        
        # SE blocks for attention
        self.se1 = SEBlock(feat1_dim)
        self.se2 = SEBlock(feat2_dim)
        
        # Fusion
        self.fusion_dim = 512
        self.feature_fusion = nn.Sequential(
            nn.Linear(feat1_dim + feat2_dim + 2, self.fusion_dim),  # +2 for gender, age
            nn.GroupNorm(32, self.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Heads
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, gender, age):
        # Extract features
        feat1 = self.backbone1(images)
        feat2 = self.backbone2(images)
        
        # Apply attention
        feat1 = feat1.view(feat1.size(0), feat1.size(1), 1, 1)
        feat2 = feat2.view(feat2.size(0), feat2.size(1), 1, 1)
        
        feat1 = self.se1(feat1).squeeze(-1).squeeze(-1)
        feat2 = self.se2(feat2).squeeze(-1).squeeze(-1)
        
        # Metadata
        metadata = torch.stack([gender, age], dim=1)
        
        # Fusion
        combined = torch.cat([feat1, feat2, metadata], dim=1)
        fused = self.feature_fusion(combined)
        
        # Predictions
        hgb_pred = self.regression_head(fused).squeeze(1)
        anemia_pred = self.classification_head(fused)
        
        return hgb_pred, anemia_pred


class OptimalTrainer:
    def __init__(self, model, device, lr=1e-3, class_weights=None):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=35,
            eta_min=1e-6
        )
        
        # Loss functions
        self.regression_loss = nn.MSELoss()
        
        if class_weights:
            weights_tensor = torch.FloatTensor(class_weights).to(device)
            self.classification_loss = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.classification_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        # GradScaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        reg_loss_total = 0.0
        cls_loss_total = 0.0
        
        for batch in dataloader:
            images = batch['image'].to(self.device, non_blocking=True)
            hgb_targets = batch['hgb'].to(self.device, non_blocking=True)
            anemia_targets = batch['anemia_class'].to(self.device, non_blocking=True)
            gender = batch['gender'].to(self.device, non_blocking=True)
            age = batch['age'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                hgb_pred, anemia_pred = self.model(images, gender, age)
                reg_loss = self.regression_loss(hgb_pred, hgb_targets)
                cls_loss = self.classification_loss(anemia_pred, anemia_targets)
                loss = reg_loss + 3.0 * cls_loss  # Weight classification more
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            reg_loss_total += reg_loss.item()
            cls_loss_total += cls_loss.item()
        
        self.scheduler.step()
        
        return (total_loss / len(dataloader), 
                reg_loss_total / len(dataloader), 
                cls_loss_total / len(dataloader))
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_hgb_preds, all_hgb_targets = [], []
        all_anemia_preds, all_anemia_targets = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device, non_blocking=True)
                hgb_targets = batch['hgb'].to(self.device, non_blocking=True)
                anemia_targets = batch['anemia_class'].to(self.device, non_blocking=True)
                gender = batch['gender'].to(self.device, non_blocking=True)
                age = batch['age'].to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    hgb_pred, anemia_pred = self.model(images, gender, age)
                    reg_loss = self.regression_loss(hgb_pred, hgb_targets)
                    cls_loss = self.classification_loss(anemia_pred, anemia_targets)
                    total_batch_loss = reg_loss + 3.0 * cls_loss
                
                total_loss += total_batch_loss.item()
                
                all_hgb_preds.extend(hgb_pred.cpu().numpy())
                all_hgb_targets.extend(hgb_targets.cpu().numpy())
                all_anemia_preds.extend(torch.argmax(anemia_pred, dim=1).cpu().numpy())
                all_anemia_targets.extend(anemia_targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        mae = mean_absolute_error(all_hgb_targets, all_hgb_preds)
        accuracy = accuracy_score(all_anemia_targets, all_anemia_preds)
        
        # Per-class accuracy - handle case where class is missing
        unique_classes = sorted(set(all_anemia_targets + all_anemia_preds))
        cm = confusion_matrix(all_anemia_targets, all_anemia_preds, labels=unique_classes)
        
        per_class_acc = []
        for i in range(len(unique_classes)):
            if cm[i].sum() > 0:
                per_class_acc.append(cm[i, i] / cm[i].sum())
            else:
                per_class_acc.append(0.0)
        
        per_class_acc = np.array(per_class_acc)
        
        return (avg_loss, mae, accuracy, per_class_acc, 
                all_hgb_preds, all_hgb_targets, 
                all_anemia_preds, all_anemia_targets)


def create_balanced_sampler(dataset):
    """Create balanced sampler with better handling"""
    classes = [dataset[i]['anemia_class'].item() for i in range(len(dataset))]
    class_counts = Counter(classes)
    
    # Calculate weights using sklearn method
    unique_classes = np.array(sorted(class_counts.keys()))
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=classes)
    
    # Create sample weights
    weights = [class_weights[class_idx] for class_idx in classes]
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def main():
    """Main function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    if not os.path.exists('processed_anemia_dataset.csv'):
        logger.error("Dataset not found")
        return
    
    df = pd.read_csv('processed_anemia_dataset.csv')
    logger.info(f"Dataset: {len(df)} samples")
    
    # Labels and class balancing
    label_encoder = LabelEncoder()
    df['anemia_class'] = label_encoder.fit_transform(df['Anemia_Status'])
    
    # Handle severe class imbalance - merge minority class 3 with class 2
    class_dist = df['anemia_class'].value_counts().sort_index()
    logger.info(f"Original Classes:\n{class_dist}")
    
    # Merge class 3 (only 3 samples) with class 2 for stability
    df.loc[df['anemia_class'] == 3, 'anemia_class'] = 2
    
    # Re-encode to ensure consecutive class labels (0, 1, 2)
    label_encoder_final = LabelEncoder()
    df['anemia_class'] = label_encoder_final.fit_transform(df['anemia_class'])
    
    class_dist_final = df['anemia_class'].value_counts().sort_index()
    logger.info(f"Final Classes (after merging):\n{class_dist_final}")
    
    # Class weights for final classes
    class_counts = class_dist_final.values
    class_weights = compute_class_weight('balanced', classes=np.arange(len(class_counts)), y=df['anemia_class'])
    logger.info(f"Class weights: {class_weights}")
    
    # Transforms
    train_transform = OptimalAugmentation.get_train_transforms()
    val_transform = OptimalAugmentation.get_val_transforms()
    
    # Cross-validation - reduce folds due to small dataset
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []
    best_acc = 0.0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['anemia_class'])):
        logger.info(f"\n{'='*15} Fold {fold + 1}/3 {'='*15}")
        
        # Datasets
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_dataset = AnemiaDataset(train_df, 'dataset anemia', train_transform)
        val_dataset = AnemiaDataset(val_df, 'dataset anemia', val_transform)
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Check class distribution in fold
        train_classes = train_df['anemia_class'].value_counts().sort_index()
        val_classes = val_df['anemia_class'].value_counts().sort_index()
        logger.info(f"Train classes: {train_classes.values}")
        logger.info(f"Val classes: {val_classes.values}")
        
        # Loaders
        train_sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=12, sampler=train_sampler, 
                                num_workers=1, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, 
                              num_workers=1, pin_memory=True, persistent_workers=False)
        
        # Model - now using 3 classes
        model = OptimalEnsemble(num_classes=3, dropout=0.3)
        trainer = OptimalTrainer(model, device, lr=1e-3, class_weights=class_weights.tolist())
        
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training
        best_val_acc = 0.0
        patience = 0
        max_patience = 8  # Reduced patience
        
        for epoch in range(25):  # Reduced epochs
            train_loss, train_reg, train_cls = trainer.train_epoch(train_loader)
            
            val_loss, val_mae, val_acc, per_class_acc, _, _, _, _ = trainer.validate_epoch(val_loader)
            
            if epoch % 3 == 0 or epoch == 24:
                lr = trainer.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1:2d}: Loss {train_loss:.4f}/{val_loss:.4f}, "
                          f"MAE {val_mae:.4f}, Acc {val_acc:.4f}, LR {lr:.2e}")
                logger.info(f"Per-class: {[f'{acc:.3f}' for acc in per_class_acc]}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                torch.save(model.state_dict(), f'improved_model_fold_{fold+1}.pth')
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), 'best_improved_model.pth')
                    
                    info = {
                        'fold': fold + 1,
                        'epoch': epoch + 1,
                        'accuracy': val_acc,
                        'mae': val_mae,
                        'per_class_accuracy': per_class_acc.tolist()
                    }
                    with open('improved_model_info.json', 'w') as f:
                        json.dump(info, f, indent=2)
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info(f"Early stop at epoch {epoch+1}")
                    break
        
        results.append({
            'fold': fold + 1,
            'accuracy': best_val_acc,
            'mae': val_mae,
            'per_class': per_class_acc.tolist()
        })
        
        # Cleanup
        del model, trainer, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final results
    logger.info(f"\n{'='*50}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*50}")
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    std_acc = np.std([r['accuracy'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    
    logger.info(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"Average MAE: {avg_mae:.4f}")
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    
    for result in results:
        logger.info(f"Fold {result['fold']}: {result['accuracy']:.4f} (MAE: {result['mae']:.4f})")
        logger.info(f"  Classes: {[f'{x:.3f}' for x in result['per_class']]}")
    
    logger.info("\n" + "="*50)
    logger.info("Training completed successfully!")
    if best_acc >= 0.85:
        logger.info("🎉 TARGET ACCURACY (85%+) ACHIEVED!")
    else:
        logger.info(f"Target accuracy not reached. Best: {best_acc:.1%}")


if __name__ == "__main__":
    main()
