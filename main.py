from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

#import models 
pipeline=joblib.load('model/pipeline.joblib')
encoder= joblib.load('model/encoder.joblib')


#create an instance of fastapi
app=FastAPI(
    title="Sepsis Classification FASTAPI"
)

#create a class
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
def predict_sepsis(sepsis_features:SepsisFeatures):
    #dataframe to hold inputs
    df = pd.DataFrame([sepsis_features.model_dump()])
    
    prediction =pipeline.predict(df)
    
    ecoded_prediction= encoder.inverse_transform([prediction])[0]
    
    prediction_output ={"prediction": ecoded_prediction }
    return prediction_output
    