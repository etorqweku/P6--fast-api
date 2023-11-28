from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

pipeline=joblib.load('model/pipeline.joblib')
encoder= joblib.load('model/encoder.joblib')
print(pipeline)
print(encoder)
app=FastAPI()

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
    return 'Hello' 

@app.get('/info')
def appinfo():
    return "info page"

@app.post('/predict_sepsis')
def predict_sepssi(sepsis_features: SepsisFeatures):
    
    df = pd.DataFrame()
    
    prediction =pipeline.predict(df)[0]
    
    return {"prediction":prediction}

@app.post('/hi')
def create():
    pass