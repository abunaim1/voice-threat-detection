"""
Prepare features for model training
Split data into train/validation/test sets

Author: abunaim1
Date: 2025-10-16
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path


class TrainingDataPreparator:
    def __init__(self, features_file='data/features.csv'):
        """Load features"""
        self.df = pd.read_csv(features_file)
        self.feature_cols = [col for col in self.df.columns if col.startswith('feature_')]
        
    def prepare_data(self, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split data into train/validation/test sets
        
        Args:
            test_size: Proportion for test set (default 15%)
            val_size: Proportion for validation set (default 15%)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train/val/test splits
        """
        print("\n" + "="*60)
        print("📦 Preparing Training Data")
        print("="*60)
        
        # Extract features and labels
        X = self.df[self.feature_cols].values
        y = self.df['label'].values
        
        # Encode labels (threatened=0, normal=1, unrecognized=2)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"\n📊 Dataset Info:")
        print(f"  Total samples: {len(X)}")
        print(f"  Features per sample: {X.shape[1]}")
        print(f"  Classes: {list(label_encoder.classes_)}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\n📂 Data Split:")
        print(f"  Training:   {len(X_train):4d} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val):4d} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:       {len(X_test):4d} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Standardize features (fit on train, transform all)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler and label encoder
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(label_encoder, 'models/label_encoder.pkl')
        
        print(f"\n💾 Saved:")
        print(f"  Scaler: models/scaler.pkl")
        print(f"  Label Encoder: models/label_encoder.pkl")
        
        # Prepare data dictionary
        data = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'scaler': scaler,
            'feature_names': self.feature_cols
        }
        
        # Save as numpy arrays
        np.savez('data/training_data.npz',
                 X_train=X_train_scaled,
                 X_val=X_val_scaled,
                 X_test=X_test_scaled,
                 y_train=y_train,
                 y_val=y_val,
                 y_test=y_test)
        
        print(f"  Training data: data/training_data.npz")
        
        print("\n" + "="*60)
        print("✅ Training Data Ready!")
        print("="*60)
        
        return data


# Usage
if __name__ == "__main__":
    preparator = TrainingDataPreparator('data/features.csv')
    data = preparator.prepare_data(
        test_size=0.15,      # 15% for testing
        val_size=0.15,       # 15% for validation
        random_state=42      # 70% for training
    )
    
    print("\n👍 Ready for Phase 4: Model Training!")