# 🧩 Step 1: Install Required Libraries
!pip install -q transformers torchaudio librosa

# 🧠 Step 2: Import Libraries
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from IPython.display import Audio

# 📥 Step 3: Upload Your Audio File
from google.colab import files
uploaded = files.upload()

# Get the uploaded file name
import os
audio_path = list(uploaded.keys())[0]

# 🎧 Step 4: Load Pretrained Wav2Vec2 Model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 🎙️ Step 5: Transcription Function
def transcribe_audio_wav2vec(audio_path):
    print("Transcribing:", audio_path)
    # Load and resample audio to 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Tokenize
    input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values
    
    # Get logits
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription.lower()

# 🔍 Step 6: Transcribe and Display Result
print("🔊 Playing Audio:")
display(Audio(audio_path))
print("📄 Transcription:")
print(transcribe_audio_wav2vec(audio_path))
