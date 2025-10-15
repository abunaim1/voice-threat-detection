"""
Data Augmentation - Preserves Emotional Tone
Works for Bangla, English, and any language.
Increases dataset size while maintaining emotion/tone.

Author: abunaim1
Date: 2025-10-15
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import audiomentations as AA


class EmotionPreservingAugmenter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Augmentations that PRESERVE emotional tone
        self.augment_pipeline = AA.Compose([
            # Light noise (simulates real-world conditions)
            AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
            
            # Small time stretch (keeps emotion intact)
            AA.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            
            # Small pitch shift (keeps emotional context)
            AA.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            
            # Slight volume change
            AA.Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.5),
        ])
    
    def load_audio(self, file_path):
        """Load and resample audio"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio
    
    def augment_audio(self, audio, n_augmentations=3):
        """Generate augmented versions of audio"""
        augmented_samples = []
        
        for i in range(n_augmentations):
            augmented = self.augment_pipeline(samples=audio, sample_rate=self.sample_rate)
            augmented_samples.append(augmented)
        
        return augmented_samples
    
    def augment_dataset(self, input_dir, output_dir, n_augmentations=3):
        """Augment entire dataset"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        classes = ['threatened', 'normal', 'unrecognized']
        
        print("\n" + "="*60)
        print(f"🔊 Starting Data Augmentation (x{n_augmentations + 1})...")
        print("="*60)
        
        for cls in classes:
            class_input = input_path / cls
            class_output = output_path / cls
            class_output.mkdir(parents=True, exist_ok=True)
            
            # Find audio files
            audio_files = list(class_input.glob('*.wav'))
            
            if len(audio_files) == 0:
                print(f"⚠️  No files found in {class_input}")
                continue
            
            print(f"\n📂 Processing {cls}: {len(audio_files)} original files")
            
            for audio_file in tqdm(audio_files, desc=f"  Augmenting {cls}"):
                try:
                    # Load original audio
                    audio = self.load_audio(audio_file)
                    
                    # Save original
                    original_output = class_output / audio_file.name
                    sf.write(original_output, audio, self.sample_rate)
                    
                    # Generate and save augmented versions
                    augmented = self.augment_audio(audio, n_augmentations)
                    
                    for i, aug_audio in enumerate(augmented):
                        aug_filename = f"{audio_file.stem}_aug{i+1}.wav"
                        aug_output = class_output / aug_filename
                        sf.write(aug_output, aug_audio, self.sample_rate)
                
                except Exception as e:
                    print(f"\n  ❌ Error augmenting {audio_file.name}: {e}")
            
            # Count total files
            total_files = len(list(class_output.glob('*.wav')))
            print(f"  ✅ {cls}: {total_files} total files ({len(audio_files)} original + {total_files - len(audio_files)} augmented)")
        
        print("\n" + "="*60)
        print("✅ Data Augmentation Complete!")
        print(f"📁 Output directory: {output_dir}")
        print("="*60)


# Usage
if __name__ == "__main__":
    augmenter = EmotionPreservingAugmenter(sample_rate=16000)
    
    print("This will create augmented versions of your audio files")
    print("Each original file will generate 3 augmented versions")
    print("Augmentations preserve emotional tone:\n")
    print("  - Add realistic background noise")
    print("  - Slight time stretching")
    print("  - Small pitch variations")
    print("  - Volume adjustments")
    
    augmenter.augment_dataset(
        input_dir="data/processed",
        output_dir="data/augmented",
        n_augmentations=3  # Creates 3 variations per original
    )