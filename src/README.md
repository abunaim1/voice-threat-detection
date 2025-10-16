# 🎤 Voice Threat Detection - Usage Guide

## 📋 Quick Start

### Test the Model

```bash
# Activate environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Run quick test
python test_model.py
```

---

## 🎯 Predict on NEW Audio Files

### Method 1: Single File Prediction

```bash
python src/predict.py path/to/your_audio.wav
```

**Example:**
```bash
python src/predict.py my_voice_samples/sample1.wav
```

**Output:**
```
============================================================
🎯 PREDICTION RESULT
============================================================
📁 File: my_voice_samples/sample1.wav
🎤 Prediction: THREATENED
📊 Confidence: 94.23%

📈 Class Probabilities:
  normal         : ██ 5.12%
  threatened     : █████████████████████████████████████ 94.23%
  unrecognized   : 0.65%
============================================================
```

---

### Method 2: Batch Prediction (Folder)

```bash
python src/predict.py path/to/folder/
```

**Example:**
```bash
python src/predict.py my_voice_samples/
```

**Output:**
```
🔄 Processing 10 files...
🔴 sample1.wav: THREATENED (94.2%)
🟢 sample2.wav: NORMAL (87.5%)
🔴 sample3.wav: THREATENED (91.3%)
🟢 sample4.wav: NORMAL (89.1%)
...

============================================================
📊 BATCH PREDICTION SUMMARY
============================================================
  Threatened     :   4 files (40.0%)
  Normal         :   5 files (50.0%)
  Unrecognized   :   1 files (10.0%)

  Total processed: 10 files
============================================================
```

---

## 🎤 Recording & Testing Your Own Voice

### Step 1: Record Audio

**Using your device or computer:**

```bash
# Linux/Mac (using sox)
rec -r 16000 -c 1 my_test.wav trim 0 3

# Or use any recording app:
# - Audacity (desktop)
# - Voice Recorder (phone)
# - Any recording device
```

**Requirements:**
- Duration: 2-5 seconds
- Format: WAV or MP3
- Any sample rate (will be converted to 16kHz)

---

### Step 2: Test Different Scenarios

**Test 1: Normal Voice**
```bash
# Record yourself speaking calmly
# "আমি ভালো আছি" or "Hello, how are you?"
python src/predict.py normal_test.wav
```

**Test 2: Threatened Voice**
```bash
# Record yourself in panicked/aggressive tone
# "আমাকে বাঁচাও!" or "Help me!" (in panic tone)
python src/predict.py threatened_test.wav
```

**Test 3: Unclear Audio**
```bash
# Record mumbling or noisy environment
python src/predict.py unclear_test.wav
```

---

## 🔧 Using in Python Code

```python
from src.predict import VoiceThreatPredictor

# Initialize predictor
predictor = VoiceThreatPredictor()

# Predict single file
result = predictor.predict('my_audio.wav')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")

# Access probabilities
for class_name, prob in result['probabilities'].items():
    print(f"{class_name}: {prob*100:.2f}%")
```

---

## 🌐 Integration with Your Device

### Option 1: Python Script

```python
import subprocess

# Record from device
audio_file = record_from_device()  # Your device's recording function

# Predict
result = subprocess.run(
    ['python', 'src/predict.py', audio_file],
    capture_output=True
)

print(result.stdout)
```

### Option 2: Real-time API (Coming soon)

---

## 📊 Understanding Results

### Prediction Classes

| Class | Description | Action |
|-------|-------------|--------|
| **THREATENED** | Panic, fear, aggression detected | 🚨 Alert/Action needed |
| **NORMAL** | Calm, conversational tone | ✅ No action |
| **UNRECOGNIZED** | Unclear audio, noise | ⚠️ Re-record/check audio |

### Confidence Score

- **> 90%**: Very confident prediction
- **70-90%**: Good confidence
- **50-70%**: Moderate confidence
- **< 50%**: Low confidence, manual review recommended

---

## 🎯 Testing Checklist

Test your model with:

- [ ] Your own calm voice (Bangla)
- [ ] Your own calm voice (English)
- [ ] Your own panicked voice (Bangla)
- [ ] Your own panicked voice (English)
- [ ] Friend/family normal conversation
- [ ] Friend/family shouting/aggressive
- [ ] Noisy environment recording
- [ ] Very quiet/distant voice
- [ ] Multiple people talking

---

## 🐛 Troubleshooting

### Error: "No trained models found!"
```bash
# Train models first
python src/train_traditional_ml.py
python src/train_deep_learning.py
```

### Error: "Could not load audio"
- Check file format (WAV/MP3)
- Check file is not corrupted
- Try converting: `ffmpeg -i input.mp3 output.wav`

### Low Accuracy on Your Voice
- Model trained on your dataset
- If testing different speakers, retrain with diverse samples
- Ensure similar recording conditions as training data

---

## 📞 Support

For issues or questions:
- Check `reports/` folder for model performance
- Review training logs
- Ensure Python 3.8+ and dependencies installed

---

**Happy Testing! 🎉**