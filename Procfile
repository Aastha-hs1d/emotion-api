web: gunicorn emotion_api.wsgi:application --worker-class=gevent --worker-connections=1000 --workers=3 --timeout 120 --bind 0.0.0.0:$PORT
