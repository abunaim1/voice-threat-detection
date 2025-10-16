"""
Voice Threat Detection - Prediction Script
Use trained model to predict on NEW audio files

Usage:
    python src/predict.py path/to/audio.wav
    python src/predict.py path/to/folder/

Author: abunaim1
Date: 2025-10-16
"""

import numpy as np
import librosa
import soundfile as sf
import joblib
from pathlib import Path
import sys
import json
from tensorflow import keras


class VoiceThreatPredictor:
    def __init__(self, model_path=None, model_type='best'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (optional)
            model_type: 'traditional' or 'deep_learning' or 'best' (auto-select)
        """
        print("\n" + "="*60)
        print("🎤 Voice Threat Detection System")
        print("="*60)
        
        # Load scaler and label encoder
        self.scaler = joblib.load('models/scaler.pkl')
        self.label_encoder = joblib.load('models/label_encoder.pkl')
        self.class_names = list(self.label_encoder.classes_)
        
        # Load model
        if model_path:
            self.model, self.model_type = self._load_model(model_path)
        else:
            self.model, self.model_type = self._load_best_model()
        
        print(f"✅ Model loaded: {self.model_type}")
        print(f"📊 Classes: {self.class_names}")
        print("="*60)
    
    def _load_model(self, model_path):
        """Load model from path"""
        model_path = Path(model_path)
        
        if model_path.suffix == '.pkl':
            # Traditional ML model
            model = joblib.load(model_path)
            return model, 'traditional'
        elif model_path.suffix in ['.h5', '.keras']:
            # Deep learning model
            model = keras.models.load_model(model_path)
            return model, 'deep_learning'
        else:
            raise ValueError(f"Unknown model format: {model_path.suffix}")
    
    def _load_best_model(self):
        """Auto-load best performing model"""
        # Check for best model metadata
        traditional_meta = Path('models/best_model_metadata.json')
        dl_meta = Path('models/best_dl_model_metadata.json')
        
        traditional_acc = 0
        dl_acc = 0
        
        if traditional_meta.exists():
            with open(traditional_meta, 'r') as f:
                data = json.load(f)
                traditional_acc = data['test_accuracy']
        
        if dl_meta.exists():
            with open(dl_meta, 'r') as f:
                data = json.load(f)
                dl_acc = data['test_accuracy']
        
        # Load best model
        if dl_acc > traditional_acc:
            # Load deep learning model
            dl_model_files = list(Path('models').glob('best_dl_model_*.h5'))
            if dl_model_files:
                return keras.models.load_model(dl_model_files[0]), 'deep_learning'
        
        # Load traditional ML model
        trad_model_files = list(Path('models').glob('best_traditional_model_*.pkl'))
        if trad_model_files:
            return joblib.load(trad_model_files[0]), 'traditional'
        
        raise FileNotFoundError("No trained models found!")
    
    def extract_features(self, audio_path):
        """Extract features from audio file (same as training)"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, duration=3.0)
        except Exception as e:
            print(f"❌ Error loading {audio_path}: {e}")
            return None
        
        features = {}
        
        # 1. MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        mfcc_delta = librosa.feature.delta(mfcc)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
        
        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 5. RMS Energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 6. Pitch
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
        
        # 7. Tempo
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        features['tempo'] = tempo[0] if len(tempo) > 0 else 0
        
        # 8. Harmonic and Percussive
        harmonic, percussive = librosa.effects.hpss(audio)
        features['harmonic_mean'] = np.mean(harmonic)
        features['harmonic_std'] = np.std(harmonic)
        features['percussive_mean'] = np.mean(percussive)
        features['percussive_std'] = np.std(percussive)
        
        # 9. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_mean'] = np.mean(mel_spec_db)
        features['mel_std'] = np.std(mel_spec_db)
        features['mel_max'] = np.max(mel_spec_db)
        features['mel_min'] = np.min(mel_spec_db)
        
        # Flatten features
        feature_list = []
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, np.ndarray):
                feature_list.extend(value.flatten())
            else:
                feature_list.append(value)
        
        return np.array(feature_list)
    
    def predict(self, audio_path, return_probabilities=True):
        """
        Predict threat level for audio file
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.extract_features(audio_path)
        
        if features is None:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        if self.model_type == 'deep_learning':
            # Deep learning model
            probabilities = self.model.predict(features_scaled, verbose=0)[0]
            predicted_class_idx = np.argmax(probabilities)
        else:
            # Traditional ML model
            predicted_class_idx = self.model.predict(features_scaled)[0]
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
            else:
                probabilities = None
        
        # Get class name
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Prepare result
        result = {
            'file': str(audio_path),
            'prediction': predicted_class,
            'confidence': float(probabilities[predicted_class_idx]) if probabilities is not None else None
        }
        
        if return_probabilities and probabilities is not None:
            result['probabilities'] = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities)
            }
        
        return result
    
    def predict_batch(self, audio_files):
        """Predict for multiple audio files"""
        results = []
        
        print(f"\n🔄 Processing {len(audio_files)} files...")
        
        for audio_file in audio_files:
            result = self.predict(audio_file)
            if result:
                results.append(result)
                
                # Print result
                emoji = "🔴" if result['prediction'] == 'threatened' else "🟢" if result['prediction'] == 'normal' else "⚪"
                conf = result['confidence'] * 100 if result['confidence'] else 0
                print(f"{emoji} {Path(audio_file).name}: {result['prediction'].upper()} ({conf:.1f}%)")
        
        return results
    
    def print_result(self, result):
        """Pretty print prediction result"""
        if not result:
            return
        
        print("\n" + "="*60)
        print("🎯 PREDICTION RESULT")
        print("="*60)
        print(f"📁 File: {result['file']}")
        print(f"🎤 Prediction: {result['prediction'].upper()}")
        
        if result['confidence']:
            print(f"📊 Confidence: {result['confidence']*100:.2f}%")
        
        if 'probabilities' in result:
            print("\n📈 Class Probabilities:")
            for class_name, prob in result['probabilities'].items():
                bar = "█" * int(prob * 40)
                print(f"  {class_name:15s}: {bar} {prob*100:.2f}%")
        
        print("="*60)


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("\n❌ Usage:")
        print("  python src/predict.py path/to/audio.wav")
        print("  python src/predict.py path/to/folder/")
        print("\nExample:")
        print("  python src/predict.py test_audio/threatened_sample.wav")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # Initialize predictor
    predictor = VoiceThreatPredictor()
    
    # Check if path is file or directory
    if input_path.is_file():
        # Single file prediction
        result = predictor.predict(input_path)
        predictor.print_result(result)
    
    elif input_path.is_dir():
        # Batch prediction
        audio_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.mp3'))
        
        if not audio_files:
            print(f"❌ No audio files found in {input_path}")
            sys.exit(1)
        
        results = predictor.predict_batch(audio_files)
        
        # Summary
        print("\n" + "="*60)
        print("📊 BATCH PREDICTION SUMMARY")
        print("="*60)
        
        predictions = [r['prediction'] for r in results]
        for class_name in ['threatened', 'normal', 'unrecognized']:
            count = predictions.count(class_name)
            percentage = (count / len(results)) * 100
            print(f"  {class_name.capitalize():15s}: {count:3d} files ({percentage:.1f}%)")
        
        print(f"\n  Total processed: {len(results)} files")
        print("="*60)
    
    else:
        print(f"❌ Path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()