"""
Compare Traditional ML vs Deep Learning Models
Select the overall best model

Author: abunaim1
Date: 2025-10-16
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load all training results"""
    # Load traditional ML results
    with open('reports/training_results.json', 'r') as f:
        traditional_results = json.load(f)
    
    # Load deep learning results (if exists)
    try:
        with open('models/best_dl_model_metadata.json', 'r') as f:
            dl_metadata = json.load(f)
        
        # Create DL results structure
        dl_results = {
            dl_metadata['model_type']: {
                'test_accuracy': dl_metadata['test_accuracy']
            }
        }
    except FileNotFoundError:
        dl_results = {}
    
    return traditional_results['models'], dl_results


def compare_models():
    """Compare all models"""
    print("\n" + "="*60)
    print("🏆 Final Model Comparison")
    print("="*60)
    
    traditional_results, dl_results = load_results()
    
    # Combine results
    all_results = {}
    
    for model_name, metrics in traditional_results.items():
        all_results[f"ML: {model_name}"] = metrics['test_accuracy']
    
    for model_name, metrics in dl_results.items():
        all_results[f"DL: {model_name}"] = metrics['test_accuracy']
    
    # Sort by accuracy
    sorted_results = dict(sorted(all_results.items(), key=lambda x: x[1], reverse=True))
    
    # Print table
    print("\n📊 Test Accuracy Ranking:\n")
    print(f"{'Rank':<6} {'Model':<30} {'Accuracy':<10}")
    print("-" * 50)
    
    for rank, (model, accuracy) in enumerate(sorted_results.items(), 1):
        print(f"{rank:<6} {model:<30} {accuracy:.4f}")
    
    # Best model
    best_model = list(sorted_results.keys())[0]
    best_accuracy = list(sorted_results.values())[0]
    
    print("\n" + "="*60)
    print("🥇 BEST MODEL")
    print("="*60)
    print(f"  Model: {best_model}")
    print(f"  Test Accuracy: {best_accuracy:.4f}")
    print("="*60)
    
    # Plot comparison
    plot_final_comparison(sorted_results)
    
    return best_model, best_accuracy


def plot_final_comparison(results):
    """Plot final comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(results.keys())
    accuracies = list(results.values())
    
    # Color code: ML vs DL
    colors = ['#4ECDC4' if 'ML:' in m else '#FF6B6B' for m in models]
    
    bars = ax.barh(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        ax.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Final Model Comparison - All Models', fontsize=16, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', label='Traditional ML'),
        Patch(facecolor='#FF6B6B', label='Deep Learning')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('reports/final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Final comparison saved: reports/final_model_comparison.png")


if __name__ == "__main__":
    best_model, best_accuracy = compare_models()