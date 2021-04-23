from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from nltk import PorterStemmer
from sklearn.linear_model import LogisticRegression
import re

filename = "Rating.pkl"
clf = pickle.load(open(filename,"rb"))
cv = pickle.load(open("transform.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get("message", False)
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',review = message,prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
