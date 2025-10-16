"""
Quick test script for voice threat detection
Test with sample audio files

Author: abunaim1
Date: 2025-10-16
"""

from src.predict import VoiceThreatPredictor
from pathlib import Path


def test_model():
    """Test model with sample files"""
    print("\n" + "="*60)
    print("🧪 Testing Voice Threat Detection Model")
    print("="*60)
    
    # Initialize predictor
    predictor = VoiceThreatPredictor()
    
    # Test with sample files from dataset
    test_files = [
        'data/augmented/threatened/threatened_001_processed_aug1.wav',
        'data/augmented/normal/normal_001_processed_aug1.wav',
        'data/augmented/unrecognized/unrecognized_001_processed_aug1.wav'
    ]
    
    # Test each file
    for test_file in test_files:
        if Path(test_file).exists():
            result = predictor.predict(test_file)
            predictor.print_result(result)
        else:
            print(f"⚠️  File not found: {test_file}")
    
    print("\n✅ Testing complete!")
    print("\n💡 To test your own audio:")
    print("   python src/predict.py your_audio.wav")


if __name__ == "__main__":
    test_model()