import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler

from flask import Flask, render_template, request

# load data
X = pd.read_csv('model/X.csv')

with open('model/y.npy', 'rb') as f:
    y = np.load(f)

# OverallQual 10
# GrLivArea 10
# GarageCars 9
# GarageArea 8
# TotalBsmtSF 7
# 1stFlrSF 6
# FullBath 5
# LotShape Reg
X = X[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'LotShape_rank']]

x_min_max_scaler = MinMaxScaler()
x_min_max_scaler.fit(X)

y_min_max_scaler = MinMaxScaler()
y_min_max_scaler.fit(y)

# load model
reconstructed_model = keras.models.load_model("model/mlp_v0.1.h5")

# run server
app = Flask(__name__, template_folder='templates')

def preprocess_data(data):
  # return np.zeros((1, 8)) # dummy data

  """
  Dictionary --> np array (1, 8)

  OverallQual 2
  GrLivArea 5000
  GarageCars 2
  GarageArea 480
  TotalBsmtSF 991
  1stFlrSF 1087
  FullBath 2
  LotShape IR3 --> 1, 2, 3, 4
  """
  X = [] # <-- OverallQual, GrLivArea, ... LotShape
  for k, v in data.items():
    if k == 'LotShape':
      if v == 'Reg':
        X.append(4)
      elif v == 'IR1':
        X.append(3)
      elif v == 'IR2':
        X.append(2)
      elif v == 'IR1':
        X.append(1)
    else:
      X.append(float(v))

  # X = [2, 5000, 2, ... , 3]
  X = np.array(X) # (8,)
  X = X.reshape((1, -1)) # (1, 8)

  # min max scaling
  scaled_X = x_min_max_scaler.transform(X)

  return scaled_X
  
@app.route("/")
def predict():
  #return "<h1>This is your Flask server.</h1>"
  return render_template("submit_form.html")

@app.route("/result", methods=['POST'])
def result():
  # Read data [v]
  # Proprocess data 
  # Model prediction
  # Return prediction

  data = request.form # User data

  message = ""
  message += "<h1>House Price</h1>"

  for k, v in data.items():
    print(k, v)
    message += k + ": " + v + "</br>"

  # 데이터 전처리
  X = preprocess_data(data) # User data --> (1, 8) array

  pred = reconstructed_model.predict(X)
  pred = y_min_max_scaler.inverse_transform(pred)
  # array (1, 1) --> string

  message += "</br>"
  message += "Predicted price: " + str(pred[0][0])

  return message
    
app.run()