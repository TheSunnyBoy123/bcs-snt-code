from xgboost import XGBClassifier
import pandas as pd
import librosa
import numpy as np
import os
from sklearn import preprocessing
from tensorflow.keras.models import load_model

xgb_loaded = XGBClassifier()
xgb_loaded.load_model("./xgb_model_weights.json")


def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    length = len(y)

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # RMS (Root Mean Square)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)

    # Harmony
    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)

    # Perceived Pitch
    perceptr = librosa.effects.percussive(y)
    perceptr_mean = np.mean(perceptr)
    perceptr_var = np.var(perceptr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)

  
    features = {
        'length': length,
        'chroma_stft_mean': chroma_stft_mean,
        'chroma_stft_var': chroma_stft_var,
        'rms_mean': rms_mean,
        'rms_var': rms_var,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_centroid_var': spectral_centroid_var,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'spectral_bandwidth_var': spectral_bandwidth_var,
        'rolloff_mean': rolloff_mean,
        'rolloff_var': rolloff_var,
        'zero_crossing_rate_mean': zero_crossing_rate_mean,
        'zero_crossing_rate_var': zero_crossing_rate_var,
        'harmony_mean': harmony_mean,
        'harmony_var': harmony_var,
        'perceptr_mean': perceptr_mean,
        'perceptr_var': perceptr_var,
        'tempo': tempo,
    }

    
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = mfcc_means[i]
        features[f'mfcc{i+1}_var'] = mfcc_vars[i]

    
    df=pd.DataFrame([features])
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
    return df

labels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal"]

def generate_predictions(model, audio_dir):
    predictions = []
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):  # or other audio formats if needed
            file_path = os.path.join(audio_dir, file_name)
            features = extract_features(file_path)
            # min_max_scaler = preprocessing.MinMaxScaler()
            # np_scaled = min_max_scaler.fit_transform(features)
            features = np.reshape(features, (1, -1))  # Reshape to match model input
            prediction = model.predict(features)
            
            predictions.append([file_name, labels[prediction[0]]])

    return predictions

# Save predictions to CSV
def save_predictions_to_csv(predictions, output_file):
    df = pd.DataFrame(predictions, columns=['Audio', 'Prediction'])
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    

    audio_dir = os.getcwd()
    # print("the directory tested is " + audio_dir)  # Make sure this is the directory where the audio files are stored
    model_path = './xgb_model_weights.json'  # Path to our trained model weights
    output_file = './prediction.csv'

    # Load the trained model
    model = xgb_loaded

    # Generate predictions
    predictions = generate_predictions(model, audio_dir)

    # Save predictions to CSV
    save_predictions_to_csv(predictions, output_file)

    print(f"Predictions saved to {output_file}")

    
