# Install required libraries
!pip install pydub noisereduce langdetect openai-whisper

# Import required libraries
import whisper # This line was causing the ModuleNotFoundError because 'whisper' was not installed.
from google.colab import files
import noisereduce as nr
from pydub import AudioSegment
import tempfile
import numpy as np
from langdetect import detect
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load Whisper model (choose 'large' for best accuracy)
model = whisper.load_model("large")

# Function to process and transcribe the audio
# Function to process and transcribe the audio
def transcribe_audio(audio_path):
    try:
        # Load the audio file (You can also use other formats like .mp3)
        audio = AudioSegment.from_file(audio_path)

        # Apply noise reduction
        audio_samples = np.array(audio.get_array_of_samples())
        audio_samples = nr.reduce_noise(y=audio_samples, sr=audio.frame_rate)

        # Save the processed audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.export(temp_file.name, format="wav")

        # Transcribe using Whisper (explicitly set language to Telugu)
        result = model.transcribe(temp_file.name, language='te')  # Telugu

        # Language detection (on transcribed text)
        detected_lang = detect(result["text"])
        logging.info(f"Detected language: {detected_lang}")

        return result["text"], detected_lang

    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        return str(e), None


# Upload the audio file
uploaded = files.upload()

# Get the uploaded audio file path
audio_file_path = list(uploaded.keys())[0]

# Run the transcription
transcription, language = transcribe_audio(audio_file_path)

# Display the transcribed text and detected language
print(f"🔹 Transcribed Text:\n{transcription}")
print(f"🔹 Detected Language: {language}")
