import os
import numpy as np
import librosa
import soundfile as sf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import traceback

# Lazy-load model to save memory on free-tier containers
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']

def get_model():
    """Load the Keras model on first request."""
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
    return model

def extract_features(file_path):
    """Extract MFCC features exactly like your working version."""
    y, sr = sf.read(file_path, dtype='float32')
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # convert to mono
    y = y[:int(sr*3)]  # limit to first 3 seconds
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc

@csrf_exempt
def predict_emotion(request):
    """Render-friendly API endpoint."""
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({'error': 'POST a .wav file with "file" field'}, status=400)

    # Save temporary file
    audio_file = request.FILES['file']
    saved_name = default_storage.save('temp.wav', audio_file)
    file_path = default_storage.path(saved_name)

    try:
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=0)

        # Predict
        model_instance = get_model()
        prediction = model_instance.predict(features, verbose=0)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        return JsonResponse({'emotion': predicted_emotion})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e) or 'Unknown error occurred'}, status=500)

    finally:
        # Always clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)
