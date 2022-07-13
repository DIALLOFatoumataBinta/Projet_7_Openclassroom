from flask import Flask, request
import pickle
import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route("/predict_target", methods=['GET'])
def predict_target():

    args = request.args
    modelpath = 'MODELS/'
    modelname = args.get("modelname")

    if modelname == 'lightGBM':
        model=pickle.load(open(modelpath+"lightGBM.pkl",'rb'))
    elif modelname == 'XGBoost':
        model=pickle.load(open(modelpath+"xgboost.pkl",'rb'))

    else:
        model=pickle.load(open(modelpath+"RandomForest.pkl",'rb'))

    path_read = 'DATA/'
    filename = "df_current_clients.csv"
    data = pd.read_csv(path_read+filename)    
    n_client=args.get("ID_client", type= int)

    #X = data[data['SK_ID_CURR'] == int(args.get("ID_client"))]
    X = data[data['SK_ID_CURR'] == int(n_client)]
    #X = data[data['SK_ID_CURR'] == int("100029")]
    X.drop(['SK_ID_CURR','TARGET'],axis=1,inplace=True)

    prediction = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0][prediction]
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba)
        }

    return dict_final

