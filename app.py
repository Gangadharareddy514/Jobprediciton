#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:42:42 2021

@author: admin
"""
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('MLP_final.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('form.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Title=""
        Location=""
        Department=""
        Description=""
        Company_profile=""
        Requirements=""
        Benefits=""
        Employment_type=""
        Required_experience=""
        Required_education=""
        Industry=""
        Function=""
        Text=""
        
        Title=re.sub(r'\W',' ',str(request.form['title']))
        Location=re.sub(r'\W',' ',str(request.form['location']))
        Department=re.sub(r'\W',' ',str(request.form['department']))
        Company_profile=re.sub(r'\W',' ',str(request.form['company_profile']))
        Description=re.sub(r'\W',' ',str(request.form['description']))
        Requirements=re.sub(r'\W',' ',str(request.form['requirements']))
        Benefits=re.sub(r'\W',' ',str(request.form['benefits']))
        Employment_type=re.sub(r'\W',' ',str(request.form['employment_type']))
        Required_experience=re.sub(r'\W',' ',str(request.form['required_experience']))
        Required_education=re.sub(r'\W',' ',str(request.form['required_education']))
        Industry=re.sub(r'\W',' ',str(request.form['industry']))
        Function=re.sub(r'\W',' ',str(request.form['function']))
        
        
        
        Text=Title+" "+Location+" "+Department+" "+Company_profile+" "+Description+" "+Requirements+" "+Benefits+" "+Employment_type+" "+Required_experience+" "+Required_education+" "+Industry+" "+Function
        #df=pd.DataFrame({"text": [Text]}, index=[0])
        #df["text"]=df["text"].values.reshape(-1,1)
        
        
        #vectorizer = CountVectorizer()
        #sentence_vectors = vectorizer.fit_transform([Text])
        #sentence_vectors=sentence_vectors.resize((17880, 104998))
        #sentence_vectors=sentence_vectors.reshape(-1, 1)
        
        #vector = CountVectorizer()
        #text_vector = vector.fit_transform(df["text"])
        #vectorizer = CountVectorizer()
        #sentence_vectors = vectorizer.fit_transform(df["text"])
        #sentence_vectors.reshape(-1,1)
        #features_test_cv = selector.transform(TfidfVectorizer.transform(vectorizer.transform(Text)))
        prediction=model.predict([Text])
        
        #output=prediction
        #prediction=model.predict([Text])
        output=prediction
        if output==0:
            return render_template('form.html',prediction_text='The Entered Job Posting is "AUTHENTIC"')
        else:
            return render_template('form.html',prediction_text='The Entered Job posting is "FRAUDULENT"')
    else:
        return render_template('form.html')

if __name__=="__main__":
    app.run(debug=True)