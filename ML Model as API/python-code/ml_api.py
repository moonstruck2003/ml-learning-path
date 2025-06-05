# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 16:43:46 2025

@author: Mridul
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle 
import json
import numpy as np

app = FastAPI()

class model_input(BaseModel):
    Pregnancies : int 
    Glucose : int 
    BloodPressure : int 
    SkinThickness : int 
    Insulin : int 
    BMI : float 
    DiabetesPedigreeFunction : float 
    Age : int 
    
#loading the model 

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

@app.post('/diabetes_prediction')

def diabetes_prediction(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    
    preg = input_dictionary["Pregnancies"]
    glu = input_dictionary["Glucose"]
    bp = input_dictionary["BloodPressure"]
    skin = input_dictionary["SkinThickness"]
    insulin = input_dictionary["Insulin"]
    bmi = input_dictionary["BMI"]
    diapefunc = input_dictionary["DiabetesPedigreeFunction"]
    age = input_dictionary["Age"]
    
    input_list = np.array([preg, glu, bp, skin, insulin, bmi, diapefunc, age]).reshape(1, -1)
    
    prediction = diabetes_model.predict(input_list)
    
    if prediction[0]==0 :
        return 'The person is not diabetic'
    else: return 'The person is daibeti'
    