from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

pipeline=joblib.load('model/pipeline.joblib')
encoder= joblib.load('model/encoder.joblib')
print(pipeline)
print(encoder)



app=FastAPI(
    title="Seppsis Classification FASTAPI"
)


class SepsisFeatures(BaseModel):
    PRG: int
    PL: int
    PR: int
    SK: int
    TS: int
    M11: float
    BD2: float
    Age: int
    Insurance: int
    
    

    
@app.get('/')
def home():
    return { "FASTAPI to classify sepssis" }

@app.get('/info')
def info():
    return 'App info page'


@app.post('/predict')
def predict_sepssi(sepsis_features:SepsisFeatures):

    df = pd.DataFrame([sepsis_features.model_dump()])
    
    prediction =pipeline.predict(df)
    
    ecoded_prediction= encoder.inverse_transform([prediction])[0]
    
    pred ={"prediction": ecoded_prediction }
    return pred
    