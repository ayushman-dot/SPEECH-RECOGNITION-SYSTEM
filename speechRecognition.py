!pip install -q transformers torchaudio librosa

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from IPython.display import Audio

from google.colab import files
uploaded = files.upload()

import os
audio_path = list(uploaded.keys())[0]

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_audio_wav2vec(audio_path):
    print("Transcribing:", audio_path)
  
    audio, sr = librosa.load(audio_path, sr=16000)
    
    input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription.lower()

print("🔊 Playing Audio:")
display(Audio(audio_path))
print("📄 Transcription:")
print(transcribe_audio_wav2vec(audio_path))
