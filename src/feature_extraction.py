"""
Feature Extraction for Voice Emotion/Tone Detection
Language Independent - Works with Bangla, English, any language
Focuses on ACOUSTIC features, NOT words

Author: abunaim1
Date: 2025-10-16
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
from tqdm import tqdm


class VoiceFeatureExtractor:
    """
    Extracts emotion/tone features from audio files.
    
    Features extracted:
    - MFCCs (voice quality)
    - Pitch (emotion indicator)
    - Energy (intensity)
    - Spectral features (voice characteristics)
    - Tempo (speech rate)
    - Zero Crossing Rate (voice stability)
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.feature_names = []
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract all features from a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector (numpy array)
        """
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=3.0)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
        
        features = {}
        
        # 1. MFCCs (Mel-frequency cepstral coefficients)
        # Captures voice quality, timbre, and vocal tract shape
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # MFCC deltas (rate of change - important for emotion)
        mfcc_delta = librosa.feature.delta(mfcc)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
        
        # 2. Chroma Features (pitch class distribution)
        # Helps detect emotional pitch patterns
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # 3. Spectral Features
        # Spectral centroid (brightness of sound)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral rolloff (frequency below which % of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # 4. Zero Crossing Rate (voice stability)
        # Higher in emotional/agitated speech
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 5. RMS Energy (voice intensity/loudness)
        # Higher in aggressive/panicked speech
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 6. Pitch Features (F0 - Fundamental Frequency)
        # CRITICAL for emotion detection
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
            features['pitch_range'] = 0
        
        # 7. Tempo/Speech Rate (speaking speed)
        # Faster in panic, slower in calm speech
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        features['tempo'] = tempo[0] if len(tempo) > 0 else 0
        
        # 8. Harmonic and Percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        features['harmonic_mean'] = np.mean(harmonic)
        features['harmonic_std'] = np.std(harmonic)
        features['percussive_mean'] = np.mean(percussive)
        features['percussive_std'] = np.std(percussive)
        
        # 9. Mel Spectrogram statistics
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_mean'] = np.mean(mel_spec_db)
        features['mel_std'] = np.std(mel_spec_db)
        features['mel_max'] = np.max(mel_spec_db)
        features['mel_min'] = np.min(mel_spec_db)
        
        # Flatten all features into single vector
        feature_vector = self._flatten_features(features)
        
        return feature_vector
    
    def _flatten_features(self, features: Dict) -> np.ndarray:
        """Flatten feature dictionary into 1D vector"""
        feature_list = []
        
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, np.ndarray):
                feature_list.extend(value.flatten())
            else:
                feature_list.append(value)
        
        return np.array(feature_list)
    
    def extract_dataset_features(self, data_dir: str, output_file: str = 'data/features.csv'):
        """
        Extract features from entire dataset.
        
        Args:
            data_dir: Directory containing class folders (threatened, normal, unrecognized)
            output_file: Where to save extracted features
            
        Returns:
            DataFrame with features and labels
        """
        data_path = Path(data_dir)
        classes = ['threatened', 'normal', 'unrecognized']
        
        all_features = []
        all_labels = []
        all_files = []
        
        print("\n" + "="*60)
        print("🔊 Extracting Features from Audio Files")
        print("="*60)
        
        for cls in classes:
            class_dir = data_path / cls
            audio_files = list(class_dir.glob('*.wav'))
            
            print(f"\n📂 Processing {cls}: {len(audio_files)} files")
            
            for audio_file in tqdm(audio_files, desc=f"  Extracting {cls}"):
                features = self.extract_features(str(audio_file))
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(cls)
                    all_files.append(audio_file.name)
        
        # Convert to DataFrame
        features_array = np.array(all_features)
        
        # Generate feature column names
        n_features = features_array.shape[1]
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        
        df = pd.DataFrame(features_array, columns=feature_columns)
        df['label'] = all_labels
        df['filename'] = all_files
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("\n" + "="*60)
        print("✅ Feature Extraction Complete!")
        print("="*60)
        print(f"📊 Total samples: {len(df)}")
        print(f"🔢 Features per sample: {n_features}")
        print(f"📁 Saved to: {output_path}")
        
        # Show class distribution
        print("\n📈 Class Distribution:")
        for cls in classes:
            count = len(df[df['label'] == cls])
            percentage = (count / len(df)) * 100
            print(f"  {cls.capitalize():15s}: {count:4d} ({percentage:.1f}%)")
        
        print("="*60)
        
        return df
    
    def get_feature_importance_names(self):
        """Get human-readable feature names for interpretation"""
        return {
            'mfcc': 'Voice Quality (MFCC)',
            'chroma': 'Pitch Pattern',
            'spectral_centroid': 'Voice Brightness',
            'spectral_rolloff': 'Frequency Distribution',
            'spectral_bandwidth': 'Frequency Range',
            'zcr': 'Voice Stability',
            'rms': 'Voice Intensity',
            'pitch': 'Fundamental Frequency',
            'tempo': 'Speech Rate',
            'harmonic': 'Harmonic Content',
            'percussive': 'Percussive Content',
            'mel': 'Mel Spectrogram'
        }


# Usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = VoiceFeatureExtractor(sample_rate=16000)
    
    # Extract features from your balanced dataset
    features_df = extractor.extract_dataset_features(
        data_dir='data/augmented',  # Your balanced dataset
        output_file='data/features.csv'
    )
    
    print("\n✅ Features ready for training!")
    print("📊 Preview:")
    print(features_df.head())