"""
Traditional Machine Learning Models
Fast training, good baseline performance

Models:
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- K-Nearest Neighbors (KNN)

Author: abunaim1
Date: 2025-10-16
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


class TraditionalMLTrainer:
    def __init__(self, data_file='data/training_data.npz'):
        """Load training data"""
        print("\n" + "="*60)
        print("🤖 Traditional ML Model Training")
        print("="*60)
        
        # Load data
        data = np.load(data_file)
        self.X_train = data['X_train']
        self.X_val = data['X_val']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_val = data['y_val']
        self.y_test = data['y_test']
        
        # Load label encoder
        self.label_encoder = joblib.load('models/label_encoder.pkl')
        self.class_names = list(self.label_encoder.classes_)
        
        print(f"\n📊 Data loaded:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Validation samples: {len(self.X_val)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Classes: {self.class_names}")
        
        self.models = {}
        self.results = {}
    
    def train_random_forest(self, n_estimators=100):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        val_acc = model.score(self.X_val, self.y_val)
        
        print(f"  ✅ Training accuracy: {train_acc:.4f}")
        print(f"  ✅ Validation accuracy: {val_acc:.4f}")
        
        return model
    
    def train_svm(self):
        """Train Support Vector Machine"""
        print("\n⚡ Training SVM...")
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True  # Enable probability estimates
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['svm'] = model
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        val_acc = model.score(self.X_val, self.y_val)
        
        print(f"  ✅ Training accuracy: {train_acc:.4f}")
        print(f"  ✅ Validation accuracy: {val_acc:.4f}")
        
        return model
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n🚀 Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = model
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        val_acc = model.score(self.X_val, self.y_val)
        
        print(f"  ✅ Training accuracy: {train_acc:.4f}")
        print(f"  ✅ Validation accuracy: {val_acc:.4f}")
        
        return model
    
    def train_knn(self, n_neighbors=5):
        """Train K-Nearest Neighbors"""
        print("\n👥 Training KNN...")
        
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['knn'] = model
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        val_acc = model.score(self.X_val, self.y_val)
        
        print(f"  ✅ Training accuracy: {train_acc:.4f}")
        print(f"  ✅ Validation accuracy: {val_acc:.4f}")
        
        return model
    
    def evaluate_model(self, model_name, model):
        """Comprehensive model evaluation"""
        print(f"\n📊 Evaluating {model_name}...")
        
        model = self.models[model_name]
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_val_pred = model.predict(self.X_val)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        results = {
            'model': model_name,
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'val_accuracy': accuracy_score(self.y_val, y_val_pred),
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'test_precision': precision_score(self.y_test, y_test_pred, average='weighted'),
            'test_recall': recall_score(self.y_test, y_test_pred, average='weighted'),
            'test_f1': f1_score(self.y_test, y_test_pred, average='weighted')
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"  Train Accuracy:  {results['train_accuracy']:.4f}")
        print(f"  Val Accuracy:    {results['val_accuracy']:.4f}")
        print(f"  Test Accuracy:   {results['test_accuracy']:.4f}")
        print(f"  Test Precision:  {results['test_precision']:.4f}")
        print(f"  Test Recall:     {results['test_recall']:.4f}")
        print(f"  Test F1 Score:   {results['test_f1']:.4f}")
        
        # Classification report
        print(f"\n  Classification Report:")
        report = classification_report(
            self.y_test, y_test_pred,
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        self._plot_confusion_matrix(
            self.y_test, y_test_pred,
            model_name,
            f'reports/confusion_matrix_{model_name}.png'
        )
        
        return results
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name, save_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Confusion matrix saved: {save_path}")
    
    def train_all_models(self):
        """Train all traditional ML models"""
        print("\n" + "="*60)
        print("🚀 Training All Traditional ML Models")
        print("="*60)
        
        # Train models
        self.train_random_forest(n_estimators=100)
        self.train_svm()
        self.train_gradient_boosting()
        self.train_knn(n_neighbors=5)
        
        print("\n" + "="*60)
        print("📊 Evaluating All Models")
        print("="*60)
        
        # Evaluate all models
        for model_name in self.models.keys():
            self.evaluate_model(model_name, self.models[model_name])
        
        # Compare models
        self._compare_models()
        
        # Save best model
        self._save_best_model()
    
    def _compare_models(self):
        """Compare all models"""
        print("\n" + "="*60)
        print("📈 Model Comparison")
        print("="*60)
        
        # Create comparison DataFrame
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.sort_values('test_accuracy', ascending=False)
        
        print("\n" + df_results.to_string())
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        models = df_results.index
        train_acc = df_results['train_accuracy']
        val_acc = df_results['val_accuracy']
        test_acc = df_results['test_accuracy']
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[0].bar(x - width, train_acc, width, label='Train', alpha=0.8)
        axes[0].bar(x, val_acc, width, label='Validation', alpha=0.8)
        axes[0].bar(x + width, test_acc, width, label='Test', alpha=0.8)
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.1])
        
        # Precision, Recall, F1
        precision = df_results['test_precision']
        recall = df_results['test_recall']
        f1 = df_results['test_f1']
        
        axes[1].bar(x - width, precision, width, label='Precision', alpha=0.8)
        axes[1].bar(x, recall, width, label='Recall', alpha=0.8)
        axes[1].bar(x + width, f1, width, label='F1 Score', alpha=0.8)
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Test Set Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Comparison plot saved: reports/model_comparison.png")
        
        # Save results to JSON
        results_json = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': self.results
        }
        
        with open('reports/training_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"✅ Results saved: reports/training_results.json")
    
    def _save_best_model(self):
        """Save the best performing model"""
        # Find best model based on test accuracy
        best_model_name = max(self.results, key=lambda x: self.results[x]['test_accuracy'])
        best_model = self.models[best_model_name]
        best_accuracy = self.results[best_model_name]['test_accuracy']
        
        print("\n" + "="*60)
        print("💾 Saving Best Model")
        print("="*60)
        print(f"  Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"  Test Accuracy: {best_accuracy:.4f}")
        
        # Save model
        model_path = f'models/best_traditional_model_{best_model_name}.pkl'
        joblib.dump(best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': best_model_name,
            'test_accuracy': float(best_accuracy),
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'classes': self.class_names,
            'n_features': self.X_train.shape[1]
        }
        
        with open('models/best_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Model saved: {model_path}")
        print(f"  ✅ Metadata saved: models/best_model_metadata.json")
        print("="*60)


# Usage
if __name__ == "__main__":
    trainer = TraditionalMLTrainer('data/training_data.npz')
    trainer.train_all_models()
    
    print("\n" + "="*60)
    print("✅ Traditional ML Training Complete!")
    print("="*60)
    print("\n📁 Check these files:")
    print("  - reports/confusion_matrix_*.png")
    print("  - reports/model_comparison.png")
    print("  - reports/training_results.json")
    print("  - models/best_traditional_model_*.pkl")
    print("\n🚀 Ready for deep learning models!")