# load all the required libraries
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from flask import Flask,jsonify,request
import joblib

# create object
app = Flask(__name__)

#  train the model
@app.route('/train_model')
def train():
    data = pd.read_excel('Historical Alarm Cases.xlsx')
    x = data.iloc[:, 1:7]
    y = data['Spuriosity Index(0/1)']
    logm = LogisticRegression()
    logm.fit(x,y)
    joblib.dump(logm,'train.pkl')
    return "Model trained successfully..."

@app.route('/test_model', methods=['POST'])
def test():
    test_data = request.get_json()
    f1 = test_data['Ambient Temperature']
    f2 = test_data['Calibration']
    f3 = test_data['Unwanted substance deposition']
    f4 = test_data['Humidity']
    f5 = test_data['H2S Content']
    f6 = test_data['detected by']
    df = pd.DataFrame(data=np.array([f1, f2, f3, f4, f5, f6]).reshape(1, 6),
                      columns=['Ambient Temperature', 'Calibration', 'Unwanted substance deposition', 'Humidity',
                                    'H2S Content', 'detected by'])
    pkl_file = joblib.load('train.pkl')
    y_pred = pkl_file.predict(df)

    if y_pred == 1:
        return "False Alarm, No Danger"
    else:
        return "True Alarm, Danger "



app.run(port=5001)


