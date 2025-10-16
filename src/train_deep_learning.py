"""
Deep Learning Models for Voice Threat Detection
Uses TensorFlow/Keras

Models:
- Feedforward Neural Network (FNN)
- 1D Convolutional Neural Network (CNN)

Author: abunaim1
Date: 2025-10-16
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


class DeepLearningTrainer:
    def __init__(self, data_file='data/training_data.npz'):
        """Load training data"""
        print("\n" + "="*60)
        print("🧠 Deep Learning Model Training")
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
        self.n_classes = len(self.class_names)
        
        # Convert labels to categorical (one-hot encoding)
        self.y_train_cat = to_categorical(self.y_train, self.n_classes)
        self.y_val_cat = to_categorical(self.y_val, self.n_classes)
        self.y_test_cat = to_categorical(self.y_test, self.n_classes)
        
        print(f"\n📊 Data loaded:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Validation samples: {len(self.X_val)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Features: {self.X_train.shape[1]}")
        print(f"  Classes: {self.class_names}")
        
        self.models = {}
        self.histories = {}
        self.results = {}
    
    def build_fnn_model(self, input_dim):
        """Build Feedforward Neural Network"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # First hidden layer
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_model(self, input_dim):
        """Build 1D CNN for feature vectors"""
        # Reshape for 1D CNN
        model = models.Sequential([
            layers.Input(shape=(input_dim, 1)),
            
            # First conv block
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Second conv block
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Third conv block
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_fnn(self, epochs=50, batch_size=32):
        """Train Feedforward Neural Network"""
        print("\n🔷 Training Feedforward Neural Network...")
        
        model = self.build_fnn_model(self.X_train.shape[1])
        
        print(f"\n📋 Model Architecture:")
        model.summary()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        history = model.fit(
            self.X_train, self.y_train_cat,
            validation_data=(self.X_val, self.y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.models['fnn'] = model
        self.histories['fnn'] = history
        
        return model, history
    
    def train_cnn(self, epochs=50, batch_size=32):
        """Train 1D CNN"""
        print("\n🔶 Training 1D Convolutional Neural Network...")
        
        # Reshape data for CNN
        X_train_cnn = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        X_val_cnn = self.X_val.reshape(self.X_val.shape[0], self.X_val.shape[1], 1)
        
        model = self.build_cnn_model(self.X_train.shape[1])
        
        print(f"\n📋 Model Architecture:")
        model.summary()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        history = model.fit(
            X_train_cnn, self.y_train_cat,
            validation_data=(X_val_cnn, self.y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.models['cnn'] = model
        self.histories['cnn'] = history
        
        return model, history
    
    def evaluate_model(self, model_name):
        """Evaluate deep learning model"""
        print(f"\n📊 Evaluating {model_name.upper()}...")
        
        model = self.models[model_name]
        
        # Prepare test data
        if model_name == 'cnn':
            X_test_eval = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        else:
            X_test_eval = self.X_test
        
        # Predictions
        y_pred_proba = model.predict(X_test_eval, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        results = {
            'model': model_name,
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'test_precision': precision_score(self.y_test, y_pred, average='weighted'),
            'test_recall': recall_score(self.y_test, y_pred, average='weighted'),
            'test_f1': f1_score(self.y_test, y_pred, average='weighted')
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"  Test Accuracy:   {results['test_accuracy']:.4f}")
        print(f"  Test Precision:  {results['test_precision']:.4f}")
        print(f"  Test Recall:     {results['test_recall']:.4f}")
        print(f"  Test F1 Score:   {results['test_f1']:.4f}")
        
        # Classification report
        print(f"\n  Classification Report:")
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        self._plot_confusion_matrix(
            self.y_test, y_pred,
            model_name,
            f'reports/confusion_matrix_dl_{model_name}.png'
        )
        
        # Plot training history
        self._plot_training_history(
            self.histories[model_name],
            model_name,
            f'reports/training_history_{model_name}.png'
        )
        
        return results
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name, save_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {model_name.upper()}',
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Confusion matrix saved: {save_path}")
    
    def _plot_training_history(self, history, model_name, save_path):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title(f'{model_name.upper()} - Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title(f'{model_name.upper()} - Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Training history saved: {save_path}")
    
    def train_all_models(self, epochs=50, batch_size=32):
        """Train all deep learning models"""
        print("\n" + "="*60)
        print("🚀 Training All Deep Learning Models")
        print("="*60)
        
        # Train FNN
        self.train_fnn(epochs=epochs, batch_size=batch_size)
        self.evaluate_model('fnn')
        
        # Train CNN
        self.train_cnn(epochs=epochs, batch_size=batch_size)
        self.evaluate_model('cnn')
        
        # Save best model
        self._save_best_model()
    
    def _save_best_model(self):
        """Save the best performing model"""
        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['test_accuracy'])
        best_model = self.models[best_model_name]
        best_accuracy = self.results[best_model_name]['test_accuracy']
        
        print("\n" + "="*60)
        print("💾 Saving Best Deep Learning Model")
        print("="*60)
        print(f"  Best Model: {best_model_name.upper()}")
        print(f"  Test Accuracy: {best_accuracy:.4f}")
        
        # Save model
        model_path = f'models/best_dl_model_{best_model_name}.h5'
        best_model.save(model_path)
        
        # Save metadata
        metadata = {
            'model_type': best_model_name,
            'test_accuracy': float(best_accuracy),
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'classes': self.class_names,
            'n_features': self.X_train.shape[1]
        }
        
        with open('models/best_dl_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Model saved: {model_path}")
        print(f"  ✅ Metadata saved: models/best_dl_model_metadata.json")
        print("="*60)


# Usage
if __name__ == "__main__":
    trainer = DeepLearningTrainer('data/training_data.npz')
    trainer.train_all_models(epochs=50, batch_size=32)
    
    print("\n" + "="*60)
    print("✅ Deep Learning Training Complete!")
    print("="*60)
    print("\n📁 Check these files:")
    print("  - reports/confusion_matrix_dl_*.png")
    print("  - reports/training_history_*.png")
    print("  - models/best_dl_model_*.h5")