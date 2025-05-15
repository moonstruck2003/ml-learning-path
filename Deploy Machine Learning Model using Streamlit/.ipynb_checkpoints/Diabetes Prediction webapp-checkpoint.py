# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:08:28 2025

@author: Mridul
"""

import numpy as np
import pickle 
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Mridul/Desktop/ML/ml-learning-path/Deploy Machine Learning Model using Streamlit/trained_model.sav', 'rb'))


#creating a function for prediction 
def DiabetesPrediction(input_data):
    
    
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    #giving a title 
    st.title('Dabetes Prediction Web App')
    
    #getting the imput data from the user
    
    Pregnancies = st.text_input("No of Pregnancies: ")
    Glucose = st.text_input("Blood glucose level: ")
    BloodPressure = st.text_input("Blood Pressure Level: ")
    SkinThickness = st.text_input("Enter Skin Thickness :")
    Insulin = st.text_input("Enter insulin level: ")
    BMI = st.text_input("Enter BMI: ")
    DiabetesPedigreeFunction = st.text_input("Enter Diabetes Predictive function: ")
    Age = st.text_input("Enter your age: ")
    
    #code for prediction 
    diagnosis = ''
    
    #creating a button for prediction 
    if st.button('Diabetes Test Result'):
        diagnosis = DiabetesPrediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()