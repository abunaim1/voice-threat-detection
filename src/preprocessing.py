"""
Audio Preprocessing - Language Independent
Works with Bangla, English, and any language.
Focuses on emotion/tone, not words.

Author: abunaim1
Date: 2025-10-15
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple
from tqdm import tqdm


class AudioPreprocessor:
    def __init__(self, target_sr=16000, target_duration=3.0):
        """
        Args:
            target_sr: Target sample rate (16kHz is standard for speech)
            target_duration: Target duration in seconds (3 seconds)
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
    
    def load_and_preprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file
        Works for ANY language (Bangla, English, etc.)
        """
        # 1. Load audio with target sample rate
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        # 2. Normalize audio amplitude (0 to 1 range)
        audio = librosa.util.normalize(audio)
        
        # 3. Remove silence from beginning and end
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # 4. Pad or truncate to target length
        if len(audio) < self.target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        else:
            # Truncate if too long
            audio = audio[:self.target_length]
        
        return audio, self.target_sr
    
    def preprocess_dataset(self, input_dir: str, output_dir: str):
        """Preprocess entire dataset"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        classes = ['threatened', 'normal', 'unrecognized']
        total_processed = 0
        
        print("\n" + "="*60)
        print("🔄 Starting Audio Preprocessing...")
        print("="*60)
        
        for cls in classes:
            class_input = input_path / cls
            class_output = output_path / cls
            class_output.mkdir(parents=True, exist_ok=True)
            
            # Find all audio files
            audio_files = list(class_input.glob('*.wav')) + list(class_input.glob('*.mp3'))
            
            if len(audio_files) == 0:
                print(f"⚠️  No files found in {class_input}")
                continue
            
            print(f"\n📂 Processing {cls}: {len(audio_files)} files")
            processed_count = 0
            error_count = 0
            
            for audio_file in tqdm(audio_files, desc=f"  Preprocessing {cls}"):
                try:
                    # Preprocess
                    audio, sr = self.load_and_preprocess(str(audio_file))
                    
                    # Save preprocessed audio
                    output_file = class_output / f"{audio_file.stem}_processed.wav"
                    sf.write(output_file, audio, sr)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"\n  ❌ Error processing {audio_file.name}: {e}")
                    error_count += 1
            
            print(f"  ✅ {cls}: {processed_count} files processed, {error_count} errors")
            total_processed += processed_count
        
        print("\n" + "="*60)
        print(f"✅ Preprocessing Complete: {total_processed} total files")
        print(f"📁 Output directory: {output_dir}")
        print("="*60)


# Usage
if __name__ == "__main__":
    preprocessor = AudioPreprocessor(target_sr=16000, target_duration=3.0)
    
    # Preprocess all data
    print("Converting all audio to standardized format:")
    print("  - Sample rate: 16kHz")
    print("  - Duration: 3 seconds")
    print("  - Channels: Mono")
    print("  - Normalized amplitude")
    
    preprocessor.preprocess_dataset(
        input_dir="data/dataset",
        output_dir="data/processed"
    )