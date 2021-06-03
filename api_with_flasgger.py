#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:27:57 2021

@author: sharad
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:53:29 2021

@author: sharad 

File for testing Docker and Falsk
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle 
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)


pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

#Entry page
@app.route('/')
def welcome():
    return "Welcome to Flask API"


#To predict by getting input from user 
# http://127.0.0.1:5000/predict?variance=0&skewness=-2&curtosis=-1&entropy=4
@app.route('/predict',methods=['GET'])
def predict():
    
    """Let's Authencita the Bank notes.
    This is using docstring for specification.
    We are using Flask for creating API, Flsagger for Frontend & Docker for contanerization
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The Output value
    
    """
    
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is "+str(prediction)


#To predict by passing a file
#use postman
@app.route('/predict_file',methods=['POST'])
def predict_file():

    """Let's Authencita the Bank notes.
    This is using docstring for specification.
    We are using Flask for creating API, Flsagger for Frontend & Docker for contanerization
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The Output values
    
    """

    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The Predicted values for the csv are "+str(list(prediction))

                                                       
if __name__== '__main__':
    app.run()