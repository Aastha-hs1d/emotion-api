# Step - 1 (Starting the Server)

- Download Python - https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
- Don't forget to select add to PATH
- Run `python -m venv .venv`, Now run, `.venv/Scripts/activate`
- Run `pip install -r requirements.txt`
- Run `cd emotion_api`
- Run `python manage.py runserver` to Start the Server

# Step - 2 (Sending Request) (Check tut.png for reference)

- Install Postman -> https://www.postman.com/downloads/
- Create New Request (HTTP)
- Now, Enter URL -> http://127.0.0.1:8000/api/predict/
- Change request type from GET -> POST
- Go to `Body` section
- Select `form-data`
- type `file` under `Key` and select the type as `file`
- under teh `Value` section, select the audio file from `Dataset` (should be .wav)
- Click on `Send`


# Important

- Model Training Notebook -> https://www.kaggle.com/code/prasoonupadhyay/emotion-analysis/edit