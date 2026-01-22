from voice_transformer import VoiceTransformer
import librosa

audio, sr = librosa.load("samples/phil/phildandy2.wav", sr=24000, mono=True)
print(f"loaded {len(audio)/sr:.1f} s mono audio @ {sr} Hz")

t = VoiceTransformer()
chars = t.analyze_voice("samples/phil/phildandy2.wav")
print("VoiceCharacteristics:")
for k, v in vars(chars).items():
    try:
        print(f"  {k}: {v:.2f}")
    except Exception:
        print(f"  {k}: {v}")
