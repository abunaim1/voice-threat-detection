"""
Data Organization Script
Helps organize and validate audio files for Voice Threat Detection.
Works with Bangla, English, and any language (emotion-based detection).

Author: abunaim1
Date: 2025-10-15
"""

import os
import shutil
from pathlib import Path
import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm


class DataOrganizer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.raw_dir = self.project_root / 'data' / 'raw'
        self.dataset_dir = self.project_root / 'data' / 'dataset'
        
        # Class directories
        self.classes = ['threatened', 'normal', 'unrecognized']
        for cls in self.classes:
            (self.dataset_dir / cls).mkdir(parents=True, exist_ok=True)
    
    def validate_audio(self, file_path):
        """Validate audio file format and properties"""
        try:
            # Load audio
            data, sr = sf.read(file_path)
            duration = len(data) / sr
            
            # Check if stereo or mono
            if len(data.shape) == 1:
                channels = 1
            else:
                channels = data.shape[1]
            
            return {
                'valid': True,
                'sample_rate': sr,
                'duration': round(duration, 2),
                'channels': channels,
                'error': None
            }
        except Exception as e:
            return {
                'valid': False,
                'sample_rate': None,
                'duration': None,
                'channels': None,
                'error': str(e)
            }
    
    def organize_files(self, source_dir):
        """
        Organize audio files from source directory
        Expected naming: {class}_{identifier}.wav
        Examples:
          - threatened_001.wav
          - normal_hello.wav
          - threatened_amake_bachaw.wav
          - normal_conversation_01.wav
        """
        source_path = Path(source_dir)
        organized_count = {'threatened': 0, 'normal': 0, 'unrecognized': 0}
        errors = []
        
        # Find all audio files
        audio_files = list(source_path.glob('*.wav')) + list(source_path.glob('*.mp3'))
        
        if len(audio_files) == 0:
            print(f"⚠️  No audio files found in {source_dir}")
            print("Please add audio files with naming: threatened_001.wav, normal_001.wav, etc.")
            return organized_count, errors
        
        print(f"Found {len(audio_files)} audio files to organize...")
        
        for audio_file in tqdm(audio_files, desc="Organizing files"):
            # Extract class from filename
            filename = audio_file.stem  # without extension
            
            # Determine class from filename
            file_class = None
            for cls in self.classes:
                if filename.lower().startswith(cls):
                    file_class = cls
                    break
            
            if file_class is None:
                errors.append(f"Could not determine class for: {audio_file.name} (should start with 'threatened_', 'normal_', or 'unrecognized_')")
                continue
            
            # Validate audio
            validation = self.validate_audio(audio_file)
            if not validation['valid']:
                errors.append(f"Invalid audio {audio_file.name}: {validation['error']}")
                continue
            
            # Copy to appropriate directory
            dest_path = self.dataset_dir / file_class / audio_file.name
            shutil.copy2(audio_file, dest_path)
            organized_count[file_class] += 1
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Organization Summary:")
        print("="*60)
        for cls, count in organized_count.items():
            print(f"  {cls.capitalize():15s}: {count:4d} files")
        
        total = sum(organized_count.values())
        print(f"  {'Total':15s}: {total:4d} files")
        
        if errors:
            print(f"\n⚠️  {len(errors)} errors occurred:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        print("="*60)
        
        return organized_count, errors
    
    def generate_dataset_report(self):
        """Generate a detailed report of the dataset"""
        report_data = []
        
        for cls in self.classes:
            class_dir = self.dataset_dir / cls
            audio_files = list(class_dir.glob('*.wav')) + list(class_dir.glob('*.mp3'))
            
            for audio_file in audio_files:
                validation = self.validate_audio(audio_file)
                if validation['valid']:
                    report_data.append({
                        'filename': audio_file.name,
                        'class': cls,
                        'sample_rate': validation['sample_rate'],
                        'duration': validation['duration'],
                        'channels': validation['channels'],
                        'path': str(audio_file)
                    })
        
        if len(report_data) == 0:
            print("⚠️  No valid audio files found in dataset directories")
            print("Please organize your files first using organize_files()")
            return None
        
        df = pd.DataFrame(report_data)
        
        # Save report
        report_path = self.project_root / 'data' / 'dataset_report.csv'
        df.to_csv(report_path, index=False)
        
        # Print statistics
        print("\n" + "="*60)
        print("📊 Dataset Statistics:")
        print("="*60)
        print(f"Total files: {len(df)}")
        
        print("\n📁 Files per class:")
        class_counts = df['class'].value_counts()
        for cls in self.classes:
            count = class_counts.get(cls, 0)
            print(f"  {cls.capitalize():15s}: {count:4d} files")
        
        print("\n⏱️  Duration statistics (seconds):")
        duration_stats = df.groupby('class')['duration'].describe()
        print(duration_stats)
        
        print("\n🎵 Sample rate distribution:")
        sr_counts = df['sample_rate'].value_counts()
        for sr, count in sr_counts.items():
            print(f"  {sr} Hz: {count} files")
        
        print("\n🔊 Channel distribution:")
        channel_counts = df['channels'].value_counts()
        for ch, count in channel_counts.items():
            ch_name = "Mono" if ch == 1 else "Stereo"
            print(f"  {ch_name}: {count} files")
        
        print(f"\n✅ Report saved to: {report_path}")
        print("="*60)
        
        return df


# Usage example
if __name__ == "__main__":
    # Initialize organizer
    organizer = DataOrganizer(project_root=".")
    
    # Step 1: Organize files from raw directory
    print("Step 1: Organizing audio files from data/raw/")
    organized_count, errors = organizer.organize_files("data/raw")
    
    # Step 2: Generate report
    if sum(organized_count.values()) > 0:
        print("\nStep 2: Generating dataset report...")
        organizer.generate_dataset_report()
    else:
        print("\n⚠️  No files were organized. Please add audio files to data/raw/")
        print("File naming format: threatened_001.wav, normal_001.wav, unrecognized_001.wav")