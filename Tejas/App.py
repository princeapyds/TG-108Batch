# load all the required libraries
from flask import Flask,jsonify,request
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# create object
app = Flask(__name__)


# create end point to  train your model and save training data in pickle file
@app.route('/train_model')
def train():
    # load data
    data = pd.read_excel('chemical_industry_data.xlsx')
    # split columns
    x = data.iloc[:, 1:7]
    y = data['Spuriosity Index(0/1)']
    # create object for Algo class
    logm = LogisticRegression()
    # train the model
    logm.fit(x, y)
    # Save trainig results in pickle file
    joblib.dump(logm, 'train.pkl')
    return "Model trained successfully"

@app.route('/test_model',methods=['POST'])
def test():
    pkl_file = joblib.load('train.pkl')
    test_data = request.get_json()
    f1 = test_data['Ambient Temperature']
    f2 = test_data['Calibration']
    f3 = test_data['Unwanted substance deposition']
    f4 = test_data['Humidity']
    f5 = test_data['H2S Content']
    f6 = test_data['detected by']
    my_test_data = [f1, f2, f3, f4, f5, f6]
    my_data_array = np.array(my_test_data)
    test_array = my_data_array.reshape(1, 6)
    df_test = pd.DataFrame(test_array,
                           columns=['Ambient Temperature', 'Calibration', 'Unwanted substance deposition', 'Humidity',
                                    'H2S Content', 'detected by'])
    y_pred = pkl_file.predict(df_test)

    if y_pred == 1:
        return "False Alarm, No Danger"
    else:
        return "True Alarm, Danger "

app.run(port=5008)

