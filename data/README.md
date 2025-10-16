# Phase 2: Data Collection Instructions

## Recording Guidelines

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

## Workflow

1. **Record** your audio files
2. **Save** them in `data/raw/` with proper naming
3. **Run** the organization script
4. **Verify** the dataset report
5. **Augment** to increase dataset size

---

## Quality Checklist

Before augmentation, ensure:
- [ ] At least 150-200 original samples per class
- [ ] Clear distinction in TONE (not words)
- [ ] Consistent recording device
- [ ] Proper file naming
- [ ] All files are playable

---

**Dataset Statistics**: python -c "from src.data_organizer import DataOrganizer; org = DataOrganizer('.'); org.dataset_dir = org.project_root / 'data' / 'augmented'; org.generate_dataset_report()"

**Remember**: The model learns EMOTION/TONE, not the actual words!
Same phrase in different tones = different classes! 🎯
