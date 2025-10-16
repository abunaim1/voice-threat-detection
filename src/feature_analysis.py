"""
Feature Analysis and Visualization
Helps understand what features distinguish threatened vs normal voice

Author: abunaim1
Date: 2025-10-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class FeatureAnalyzer:
    def __init__(self, features_file='data/features.csv'):
        """Load extracted features"""
        self.df = pd.read_csv(features_file)
        self.feature_cols = [col for col in self.df.columns if col.startswith('feature_')]
        self.X = self.df[self.feature_cols].values
        self.y = self.df['label'].values
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
    
    def plot_class_distribution(self, save_path='reports/class_distribution.png'):
        """Plot class distribution"""
        plt.figure(figsize=(10, 6))
        
        class_counts = self.df['label'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        plt.bar(class_counts.index, class_counts.values, color=colors, alpha=0.8)
        plt.title('Dataset Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(fontsize=11)
        
        # Add count labels on bars
        for i, (cls, count) in enumerate(class_counts.items()):
            plt.text(i, count + 10, str(count), ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_feature_distributions(self, save_path='reports/feature_distributions.png'):
        """Plot distribution of key features across classes"""
        # Select a few key features to visualize
        key_features = self.feature_cols[:6]  # First 6 features
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(key_features):
            for label in self.df['label'].unique():
                data = self.df[self.df['label'] == label][feature]
                axes[idx].hist(data, alpha=0.6, label=label, bins=30)
            
            axes[idx].set_title(f'{feature}', fontsize=10)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.suptitle('Feature Distributions Across Classes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_pca_2d(self, save_path='reports/pca_2d.png'):
        """2D PCA visualization of features"""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        plt.figure(figsize=(12, 8))
        
        colors = {'threatened': '#FF6B6B', 'normal': '#4ECDC4', 'unrecognized': '#95E1D3'}
        
        for label in np.unique(self.y):
            mask = self.y == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[label], label=label.capitalize(), 
                       alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        plt.title('PCA 2D Projection of Audio Features', fontsize=16, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
        print(f"\n📊 PCA Analysis:")
        print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
        print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
        print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    
    def plot_tsne_2d(self, save_path='reports/tsne_2d.png'):
        """t-SNE visualization (better for non-linear patterns)"""
        print("\n🔄 Computing t-SNE (this may take a minute)...")
        
        # Use smaller sample for faster computation if dataset is large
        if len(self.X_scaled) > 1000:
            indices = np.random.choice(len(self.X_scaled), 1000, replace=False)
            X_sample = self.X_scaled[indices]
            y_sample = self.y[indices]
        else:
            X_sample = self.X_scaled
            y_sample = self.y
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        
        plt.figure(figsize=(12, 8))
        
        colors = {'threatened': '#FF6B6B', 'normal': '#4ECDC4', 'unrecognized': '#95E1D3'}
        
        for label in np.unique(y_sample):
            mask = y_sample == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[label], label=label.capitalize(), 
                       alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        plt.title('t-SNE 2D Projection of Audio Features', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_correlation_matrix(self, save_path='reports/correlation_matrix.png'):
        """Plot correlation between features"""
        # Use subset of features for readability
        feature_subset = self.feature_cols[:20]
        corr_matrix = self.df[feature_subset].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def generate_full_report(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("📊 Generating Feature Analysis Report")
        print("="*60)
        
        self.plot_class_distribution()
        self.plot_feature_distributions()
        self.plot_pca_2d()
        self.plot_tsne_2d()
        self.plot_correlation_matrix()
        
        print("\n" + "="*60)
        print("✅ Analysis Complete!")
        print("📁 Check 'reports/' folder for visualizations")
        print("="*60)


# Usage
if __name__ == "__main__":
    analyzer = FeatureAnalyzer('data/features.csv')
    analyzer.generate_full_report() 