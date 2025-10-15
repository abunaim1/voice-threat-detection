"""
Quick Setup Helper
Creates necessary directories and placeholder files.

Author: abunaim1
Date: 2025-10-15
"""

from pathlib import Path


def create_directory_structure():
    """Create all necessary directories"""
    
    directories = [
        'data/raw',
        'data/dataset/threatened',
        'data/dataset/normal',
        'data/dataset/unrecognized',
        'data/processed/threatened',
        'data/processed/normal',
        'data/processed/unrecognized',
        'data/augmented/threatened',
        'data/augmented/normal',
        'data/augmented/unrecognized',
        'notebooks',
        'models',
        'src',
        'tests'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    
    # Create .gitkeep files in data directories
    data_dirs = ['data/raw', 'data/dataset', 'data/processed', 'data/augmented']
    for data_dir in data_dirs:
        gitkeep = Path(data_dir) / '.gitkeep'
        gitkeep.touch()
    
    print("\n✅ Directory structure created successfully!")
    print("\n📝 Next steps:")
    print("  1. Place your audio files in data/raw/")
    print("  2. Name them: threatened_001.wav, normal_001.wav, etc.")
    print("  3. Run: python src/data_organizer.py")


def create_readme_instructions():
    """Create README with recording instructions"""
    
    readme_content = """# Phase 2: Data Collection Instructions

## 📝 Recording Guidelines

### Equipment Needed
- Your voice recording device
- Quiet environment (for normal/threatened recordings)
- Noisy environment (for unrecognized recordings)

### Recording Requirements
- **Format**: WAV preferred (MP3 also works)
- **Duration**: 2-5 seconds per sample
- **Quality**: Clear audio from your device

---

## 🎤 What to Record

### 1. THREATENED Class (Panic/Fear/Aggression Tone)
Record these with **emotional intensity** (panicked, fearful, or aggressive tone):

**Bangla Examples:**
- "আমাকে বাঁচাও!" (Amake bachaw!) - in panic tone
- "সাহায্য!" (Shahajjo!) - in fearful tone
- "থামো!" (Thamo!) - in aggressive tone
- "দূরে যাও!" (Dure jao!) - in angry tone

**English Examples:**
- "Help me!" - in panic tone
- "Stop it!" - in aggressive tone
- "Leave me alone!" - in fearful tone
- "Get away!" - in angry tone

**Target**: 200+ original samples

---

### 2. NORMAL Class (Calm/Conversational Tone)
Record these in **calm, relaxed, conversational tone**:

**Bangla Examples:**
- "আমি ভালো আছি" (Ami bhalo achi) - calm
- "কেমন আছেন?" (Kemon achen?) - friendly
- "ধন্যবাদ" (Dhonnobad) - polite
- "আজ আবহাওয়া ভালো" (Aj abohawa bhalo) - casual

**English Examples:**
- "How are you?" - friendly
- "I'm fine, thank you" - calm
- "Have a nice day" - polite
- "The weather is nice" - casual

**Target**: 200+ original samples

---

### 3. UNRECOGNIZED Class (Unclear/Noisy)
Record these with **poor quality/unclear audio**:

**Examples:**
- Mumbling/unclear speech
- Speech with loud background noise
- Very quiet/distant voice
- Multiple people talking at once
- Static/distorted audio

**Target**: 200+ original samples

---

## 📁 File Naming Convention

Use this format: `{class}_{identifier}.wav`

**Examples:**

---

## 🔄 Workflow

1. **Record** your audio files
2. **Save** them in `data/raw/` with proper naming
3. **Run** the organization script
4. **Verify** the dataset report
5. **Augment** to increase dataset size

---

## ✅ Quality Checklist

Before augmentation, ensure:
- [ ] At least 150-200 original samples per class
- [ ] Clear distinction in TONE (not words)
- [ ] Consistent recording device
- [ ] Proper file naming
- [ ] All files are playable

---

**Remember**: The model learns EMOTION/TONE, not the actual words!
Same phrase in different tones = different classes! 🎯
"""
    
    readme_path = Path('data/RECORDING_INSTRUCTIONS.md')
    readme_path.write_text(readme_content, encoding='utf-8')
    print(f"✅ Recording instructions created: {readme_path}")


if __name__ == "__main__":
    create_directory_structure()
    create_readme_instructions()
    
    print("\n" + "="*60)
    print("🎉 Phase 2 setup complete!")
    print("="*60)
    print("\n📖 Read: data/RECORDING_INSTRUCTIONS.md")
    print("🎤 Start recording and place files in data/raw/")