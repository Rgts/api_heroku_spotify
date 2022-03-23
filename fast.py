from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Added
import joblib
import pandas as pd

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# define a root `/` endpoint
@app.get("/")
def index():
    return {"Status": "Up and running"}


# Implement a /predict endpoint


@app.get("/predict")
async def predict(acousticness,
                    danceability,
                    duration_ms,
                    energy,
                    explicit,
                    id,
                    instrumentalness,
                    key,
                    liveness,
                    loudness,
                    mode,
                    name,
                    release_date,
                    speechiness,
                    tempo,
                    valence,
                    artist):



    model = joblib.load("model.joblib")




    # acousticness=0.654
    # danceability=0.499
    # duration_ms=219827
    # energy=0.19
    # explicit=0
    # id="0B6BeEUd6UwFlbsHMQKjob"
    # instrumentalness=0.00409
    # key=7#
    # liveness=0.0898
    # loudness=-16.435
    # mode=1
    # name="Back in the Goodle Days"
    # release_date=1971#
    # speechiness=0.0454
    # tempo=149.46
    # valence=0.43
    # artist="John Hartford"

    X = pd.DataFrame([{
        "acousticness":float(acousticness),
        "danceability":float(danceability),
        "duration_ms":int(duration_ms),
        "energy":float(energy),
        "explicit":int(explicit),
        "id":str(id),
        "instrumentalness":float(instrumentalness),
        "key":int(key),
        "liveness":float(liveness),
        "loudness":float(loudness),
        "mode":int(mode),
        "name":str(name),
        "release_date":str(release_date),
        "speechiness":float(speechiness),
        "tempo":float(tempo),
        "valence":float(valence),
        "artist":str(artist)
        }])

    prediction = model.predict(X)  # return an number in a array
    prediction = prediction[0]  # return the number
    prediction = int(prediction)  # cast to integer


    return {
        "artist": str(artist),
        "name": str(name),
        "gross_revenue_prediction": prediction
    }



# http://127.0.0.1:8000/predict?acousticness=0.654&danceability=0.499&duration_ms=219827&energy=0.19&explicit=0&id=0B6BeEUd6UwFlbsHMQKjob&instrumentalness=0.00409&key=7&liveness=0.0898&loudness=-16.435&mode=1&name=Back%20in%20the%20Goodle%20Days&release_date=1971&speechiness=0.0454&tempo=149.46&valence=0.43&artist=John%20Hartford
