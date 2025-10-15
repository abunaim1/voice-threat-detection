# Voice Threat Detection System

An ML-based system to classify voice recordings as **threatened**, **normal**, or **unrecognized**.

## 🎯 Project Goal
Analyze audio from a device and predict whether the voice indicates a threat or normal conversation.

## 📁 Project Structure
```
voice-threat-detection/
├── data/
│   ├── raw/              # Original audio files
│   ├── processed/        # Preprocessed audio
│   └── dataset/          # Organized by class
│       ├── threatened/
│       ├── normal/
│       └── unrecognized/
├── notebooks/            # Jupyter notebooks for exploration
├── models/              # Saved trained models
├── src/                 # Source code
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── tests/               # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Setup

1. **Clone the repository** (after creating it on GitHub)
```bash
git clone https://github.com/abunaim1/voice-threat-detection.git
cd voice-threat-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset
- **Classes**: threatened, normal, unrecognized
- **Format**: WAV files (16kHz, mono)
- **Minimum**: 500-1000 samples per class

## 🔧 Usage
(Will be updated as we develop the project)

## 📈 Current Status
- [x] Phase 1: Project Setup
- [ ] Phase 2: Data Collection
- [ ] Phase 3: Feature Engineering
- [ ] Phase 4: Model Development
- [ ] Phase 5: Training & Evaluation
- [ ] Phase 6: Deployment

## 👤 Author
**abunaim1**

## 📝 License
MIT License

---
*Created: 2025-10-15*
```