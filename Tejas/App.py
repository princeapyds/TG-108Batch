from flask import Flask,jsonify,request
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/train_model')
def train():
    data = pd.read_excel('Historical Alarm Cases.xlsx')
    x = data.iloc[:, 1:7]
    y = data['Spuriosity Index(0/1)']
    logm = LogisticRegression()
    logm.fit(x, y)
    joblib.dump(logm, 'train.pkl')
    return "Model trained successfully"



app.run(port=5009)

