# SPEECH-RECOGNITION-SYSTEM

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: AYUSHMAN JAISWAL

*INTERN ID*: CT04DG3348

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION* : 4 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION*

This project demonstrates a basic yet powerful speech-to-text (STT) system that utilizes pre-trained deep learning models to transcribe spoken audio into written text. The system is built using the Wav2Vec2.0 model developed by Facebook AI, which is known for its excellent performance in automatic speech recognition (ASR) tasks. The approach here leverages the transformers library by Hugging Face, along with torchaudio and librosa, to process and transcribe audio input. This method is particularly useful for transcribing short audio clips and can be deployed in environments like Google Colab for ease of use, accessibility, and low computational overhead on the user's machine.

The process begins with uploading a short audio file, typically in .wav or .mp3 format. The audio file is then loaded and resampled to a sample rate of 16kHz using the librosa library, which is the standard sampling rate expected by the Wav2Vec2 model. Once the audio signal is prepared, it is tokenized using the Wav2Vec2 tokenizer, converting the waveform into a format the model can understand. The pre-trained model (facebook/wav2vec2-base-960h) then processes the input and outputs logits, which are essentially the model’s predictions for each possible character or word token at every time step in the audio. These logits are decoded into actual text using the argmax operation to choose the most likely token and then converting those tokens into readable words using the tokenizer’s decoding function.

This system provides a straightforward yet powerful demonstration of how modern ASR can be performed using open-source tools. Unlike traditional speech recognition systems that require hand-crafted features and complex language models, Wav2Vec2 leverages self-supervised learning and has been trained on thousands of hours of unlabeled speech, making it highly generalizable and effective even with relatively small datasets or simple implementations. The model can accurately recognize and transcribe clear and moderately noisy speech, which makes it ideal for short clips or voice notes.

One of the key advantages of using this pre-trained model is that users do not need to train their own ASR system, which can be time-consuming and resource-intensive. Instead, the pretrained Wav2Vec2 model can be downloaded and used out of the box for high-quality speech transcription. Moreover, this system can be easily extended to support real-time transcription, voice-command applications, or even integrated into mobile or web apps with some modifications. Additionally, since the system runs efficiently in a Jupyter or Colab environment, it is highly accessible to students, researchers, and developers for experimentation and learning.


