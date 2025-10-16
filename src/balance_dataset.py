"""
Dataset Balancing Script
Balances classes to have equal number of samples.

Author: abunaim1
Date: 2025-10-16
"""

import shutil
from pathlib import Path
import random
from collections import Counter


class DatasetBalancer:
    def __init__(self, data_dir='data/augmented'):
        self.data_dir = Path(data_dir)
        self.classes = ['threatened', 'normal', 'unrecognized']
    
    def count_files(self):
        """Count files in each class"""
        counts = {}
        for cls in self.classes:
            class_dir = self.data_dir / cls
            files = list(class_dir.glob('*.wav'))
            counts[cls] = len(files)
        return counts
    
    def balance_by_undersampling(self, output_dir='data/balanced'):
        """
        Balance dataset by reducing to minimum class size
        (Keeps all data quality, reduces dataset size)
        """
        output_path = Path(output_dir)
        
        # Count current files
        counts = self.count_files()
        min_count = min(counts.values())
        
        print("\n" + "="*60)
        print("⚖️  Balancing Dataset by Undersampling")
        print("="*60)
        print("\nCurrent class distribution:")
        for cls, count in counts.items():
            print(f"  {cls.capitalize():15s}: {count:4d} files")
        
        print(f"\n🎯 Target count per class: {min_count}")
        
        # Balance each class
        for cls in self.classes:
            class_input = self.data_dir / cls
            class_output = output_path / cls
            class_output.mkdir(parents=True, exist_ok=True)
            
            # Get all files
            all_files = list(class_input.glob('*.wav'))
            
            # Randomly sample min_count files
            random.shuffle(all_files)
            selected_files = all_files[:min_count]
            
            # Copy selected files
            for file in selected_files:
                shutil.copy2(file, class_output / file.name)
            
            print(f"  ✅ {cls.capitalize():15s}: {len(selected_files)} files copied")
        
        print("\n" + "="*60)
        print(f"✅ Balanced dataset created in: {output_dir}")
        print(f"📊 Total files: {min_count * len(self.classes)}")
        print("="*60)
    
    def balance_by_oversampling(self, output_dir='data/balanced'):
        """
        Balance dataset by duplicating minority classes
        (Increases dataset size, may cause overfitting)
        """
        output_path = Path(output_dir)
        
        # Count current files
        counts = self.count_files()
        max_count = max(counts.values())
        
        print("\n" + "="*60)
        print("⚖️  Balancing Dataset by Oversampling")
        print("="*60)
        print("\nCurrent class distribution:")
        for cls, count in counts.items():
            print(f"  {cls.capitalize():15s}: {count:4d} files")
        
        print(f"\n🎯 Target count per class: {max_count}")
        
        # Balance each class
        for cls in self.classes:
            class_input = self.data_dir / cls
            class_output = output_path / cls
            class_output.mkdir(parents=True, exist_ok=True)
            
            # Get all files
            all_files = list(class_input.glob('*.wav'))
            
            # Copy all original files
            for file in all_files:
                shutil.copy2(file, class_output / file.name)
            
            # Duplicate files to reach max_count
            current_count = len(all_files)
            if current_count < max_count:
                needed = max_count - current_count
                # Randomly duplicate files
                duplicates = random.choices(all_files, k=needed)
                
                for i, file in enumerate(duplicates):
                    new_name = f"{file.stem}_dup{i}.wav"
                    shutil.copy2(file, class_output / new_name)
            
            final_count = len(list(class_output.glob('*.wav')))
            print(f"  ✅ {cls.capitalize():15s}: {final_count} files")
        
        print("\n" + "="*60)
        print(f"✅ Balanced dataset created in: {output_dir}")
        print(f"📊 Total files: {max_count * len(self.classes)}")
        print("="*60)
    
    def check_balance(self, directory='data/augmented'):
        """Check if dataset is balanced"""
        self.data_dir = Path(directory)
        counts = self.count_files()
        
        min_count = min(counts.values())
        max_count = max(counts.values())
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print("\n" + "="*60)
        print("📊 Dataset Balance Check")
        print("="*60)
        
        for cls, count in counts.items():
            percentage = (count / sum(counts.values())) * 100
            print(f"  {cls.capitalize():15s}: {count:4d} files ({percentage:.1f}%)")
        
        print(f"\n📈 Imbalance Ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio <= 1.2:
            print("✅ Status: WELL BALANCED (ratio ≤ 1.2)")
            print("👍 You can proceed to Phase 3!")
        elif imbalance_ratio <= 1.5:
            print("⚠️  Status: SLIGHTLY IMBALANCED (ratio ≤ 1.5)")
            print("💡 Recommended: Balance before Phase 3 (optional)")
        else:
            print("❌ Status: IMBALANCED (ratio > 1.5)")
            print("⚠️  Required: Balance before Phase 3!")
        
        print("="*60)
        
        return imbalance_ratio


# Usage
if __name__ == "__main__":
    balancer = DatasetBalancer('data/augmented')
    
    # Check current balance
    ratio = balancer.check_balance()
    
    # If imbalanced, offer solutions
    if ratio > 1.2:
        print("\n" + "="*60)
        print("🔧 Balancing Options:")
        print("="*60)
        print("\n1. Undersampling (Recommended)")
        print("   - Reduces to smallest class size")
        print("   - Pro: No duplicate data, faster training")
        print("   - Con: Loses some data")
        
        print("\n2. Oversampling")
        print("   - Duplicates to largest class size")
        print("   - Pro: Keeps all data")
        print("   - Con: May cause overfitting")
        
        choice = input("\nChoose method (1/2) or skip (s): ").strip()
        
        if choice == '1':
            balancer.balance_by_undersampling('data/balanced')
            print("\n✅ Use 'data/balanced' directory for Phase 3")
        elif choice == '2':
            balancer.balance_by_oversampling('data/balanced')
            print("\n✅ Use 'data/balanced' directory for Phase 3")
        else:
            print("\n⏭️  Skipping balancing. Proceeding with current data.")
    else:
        print("\n✅ Your dataset is already well balanced!")
        print("👍 Ready for Phase 3!")