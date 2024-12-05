from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import librosa

# Preprocessing Function
def preprocess_audio(file_path, n_mels=128, duration=5, sr=22050):
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        if len(audio) < sr * duration:
            padding = sr * duration - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            audio = audio[:sr * duration]

        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        normalized_spectrogram = (spectrogram_db - np.mean(spectrogram_db)) / np.std(spectrogram_db)
        return normalized_spectrogram[..., np.newaxis]  # Add channel dimension
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prediction Function
def predict_audio(file_path, model, label_encoder):
    spectrogram = preprocess_audio(file_path)
    if spectrogram is None:
        print("Error in preprocessing the audio.")
        return None

    spectrogram = np.expand_dims(spectrogram, axis=0)
    predictions = model.predict(spectrogram)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_class_label

# Load Label Encoder
metadata_path = 'C:/Users/user/Desktop/DL/ESC-50-master/meta/esc50.csv'
labels_df = pd.read_csv(metadata_path)
label_encoder = LabelEncoder()
label_encoder.fit(labels_df['category'])

# Load the Trained Model
model = load_model('cnn_audio_classifier.h5')

# Test Audio File Path
test_audio_path = 'C:/Users/user/Desktop/DL/ESC-50-master/audio/1-17092-B-27.wav'

# Predict
predicted_label = predict_audio(test_audio_path, model, label_encoder)
if predicted_label:
    print(f"The predicted class for the audio file is: {predicted_label}")