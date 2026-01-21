"""
Ultra-minimal Voice Forge - The Core Logic
This is what forge.py does under the hood
"""

import torch
import torchaudio
from pathlib import Path
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

def mint_voice_simple(sample_path: Path, output_path: Path):
    """Mint a voice with zero overhead"""
    
    # Load model
    config = XttsConfig()
    config.load_json("/path/to/xtts/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
    
    # Generate conditioning (THE CRITICAL PART)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=str(sample_path),
        max_ref_length=90,  # Your 60+ second samples
        gpt_cond_len=12       # THE MAGIC NUMBER - DO NOT CHANGE
    )
    
    # Save
    torch.save({
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding
    }, output_path)

# That's it. That's the core.
