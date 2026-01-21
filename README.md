# Voice Forge 2.0 - Broadcast Grade Voice System

## Quick Start (3 Steps)

### Step 1: Record Your Samples
Create 45-90 second clean recordings for each character:

**Phil** (`samples/phil/voice_sample.wav`):
- Dry, observational humor
- Deliberate pacing
- Slightly gravelly, warm baritone

**Jim** (`samples/jim/voice_sample.wav`):
- High energy, enthusiastic
- Quick-witted, punchy delivery
- Clear tenor with variety

**Bryan** (`samples/bryan/voice_sample.wav`):
- Intense, earnest, authoritative
- Deep bass-baritone
- Builds to powerful points

**Technical specs:**
- Format: WAV
- Sample rate: 24000 Hz (will auto-convert)
- Channels: Mono preferred (will auto-convert)
- Length: 45-90 seconds
- Quality: Clean, no background noise

### Step 2: Mint the Voices
```bash
python forge.py
```

This creates permanent embeddings in `embeddings/`:
- `phil_voice.pt` - Phil's identity
- `jim_voice.pt` - Jim's identity
- `bryan_voice.pt` - Bryan's identity

**These are your voice assets. Back them up.**

### Step 3: Test & Use

**Test a character:**
```bash
python synthesize.py test phil
# Generates test phrases and saves to output/test_report_phil.json
```

**Generate specific text:**
```bash
python synthesize.py phil "Your custom text here"
# Creates: output/generated/phil_[timestamp]_[text].wav
```

**Batch generate:**
```python
from synthesize import batch_synthesize

lines = [
    "First line of dialogue",
    "Second line of dialogue",
    "Third line with different emotion"
]

paths = batch_synthesize("phil", lines)
```

## Integration with Dandy Show

Copy embeddings to your show directory:
```bash
cp embeddings/*_voice.pt ../phil_and_jim_dandy_show/voices/
```

Then in your show script:
```python
from voice_forge_v2.synthesize import VoiceSynthesizer

synth = VoiceSynthesizer()
synth.synthesize("phil", script_line)
```

## Voice Quality Checklist

After minting, test with these phrases:

1. **Consistency Test:** Generate same line 3 times - should sound identical
2. **Emotion Test:** Generate serious line, then funny line - should maintain character
3. **Length Test:** Generate 15-second line - should not drift or degrade
4. **Breathing Test:** Long sentences should sound natural, not rushed

## Troubleshooting

**"Sounds robotic":**
- Source recording quality issue
- Sample too short (<30 seconds)
- Too much background noise

**"Inconsistent between generations":**
- Temperature too high (>0.7)
- GPT cond length wrong (should be 12)

**"Doesn't sound like character":**
- Source recording doesn't match profile
- Re-record following personality guidelines

## Version & Stability

- **Version:** 2.0.0
- **Deterministic:** Yes (same input = same output)
- **Stable embeddings:** Mint once, use forever
- **Model:** XTTS v2 (frozen)

Never upgrade XTTS version. These settings are frozen for broadcast consistency.

---

Repository: https://github.com/Spruked/Voice_Forge_2.0.git

License
-------
This project is released under the MIT License. See the `LICENSE` file for details.

.gitignore
--------
A `.gitignore` has been added to exclude large artifacts (models, embeddings, generated audio, and raw sample WAVs). Adjust as needed before pushing.
