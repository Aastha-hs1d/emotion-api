import os
import numpy as np
import librosa
import soundfile as sf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from django.core.files.storage import default_storage
import traceback  # Add this import

# Load model once
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
model = load_model(MODEL_PATH)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']


def extract_features(file_path):
    
    """
    Extracts MFCC (Mel-frequency cepstral coefficients) features from the input audio file.

    Parameters:
    -----------
    file_path : str
        Path to the .wav audio file.

    Returns:
    --------
    numpy.ndarray
        Extracted MFCC features with shape (40, 1), ready to be passed into the model.

    Notes:
    ------
    - Uses a fixed duration of 3 seconds with an offset of 0.5s for consistency.
    - MFCCs are a widely-used feature representation in audio and speech classification tasks.
    - Applies mean aggregation along the time axis for simplicity.
    """

    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)  # (40, 1)
    return mfcc

@csrf_exempt
def predict_emotion(request):

    """
    Django view to handle emotion prediction API requests.

    Method: POST
    Endpoint: /api/predict/

    Expects:
    --------
    - A POST request with a .wav file under the "file" key in the multipart form.

    Returns:
    --------
    - JsonResponse containing:
        {'emotion': predicted_label} on success
        {'error': error_message} on failure

    Notes:
    ------
    - CSRF is exempted to allow API testing via tools like Postman or curl.
    - Temporarily saves the uploaded audio file for feature extraction.
    - Automatically deletes the temp file after prediction.
    """

    if request.method == 'POST' and request.FILES.get('file'):
        audio_file = request.FILES['file']
        saved_name = default_storage.save('temp.wav', audio_file)
        file_path = default_storage.path(saved_name)

        try:
            features = extract_features(file_path)
            print(f"[INFO] Extracted features shape: {features.shape}")

            features = np.expand_dims(features, axis=0)
            prediction = model.predict(features)
            print(prediction)
            predicted_emotion = emotion_labels[np.argmax(prediction)]

            os.remove(file_path)
            return JsonResponse({'emotion': predicted_emotion})

        except Exception as e:
            print("[ERROR] An exception occurred:")
            traceback.print_exc()  # ✅ Logs full traceback to console
            return JsonResponse({'error': str(e) or 'Unknown error occurred'}, status=500)

    return JsonResponse({'error': 'POST a .wav file with "file" field'}, status=400)
