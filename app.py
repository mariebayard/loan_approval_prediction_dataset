from flask import Flask, render_template, request
from flask_paginate import Pagination, get_page_args
import joblib
import numpy as np
import pandas as pd

app=Flask(__name__)

df=pd.read_csv('loan_approval_dataset.csv')
model= joblib.load('tf_model01.pkl')
model_rf=joblib.load('best_random_forest_model.pkl')
model_XGB=joblib.load('best_xgboost_model.pkl')
model_dt=joblib.load('best_decision_tree_model.pkl')

scaler=joblib.load('minmax_scaler.pkl')

@app.route('/')
def firstpage():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    page, per_page, offset = get_page_args(per_page=100)
    total = len(df)
    pagination_data = df.iloc[offset: offset + per_page]
    pagination = Pagination(page=page, per_page=per_page, total=total)
    return render_template('dataset.html', tables=pagination_data.to_html(classes='table table-striped'), pagination=pagination)

def get_features(request):
    no_of_dependants = int(request.form['no_of_dependants'])
    education = int(request.form['education'])
    self_employed = int(request.form['self_employed'])
    income_annum = float(request.form['income_annum'])
    loan_amount = float(request.form['loan_amount'])
    loan_term = float(request.form['loan_term'])
    cibil_score = float(request.form['cibil_score'])
    residential_assets_value = float(request.form['residential_assets_value'])
    commercial_assets_value = float(request.form['commercial_assets_value'])
    luxury_assets_value = float(request.form['luxury_assets_value'])
    bank_asset_value = float(request.form['bank_asset_value'])

    return np.array([[no_of_dependants, education, self_employed, income_annum, loan_amount, loan_term,
                      cibil_score, residential_assets_value, commercial_assets_value,
                      luxury_assets_value, bank_asset_value]])
def make_prediction_tf(model, features):
    prediction = model.predict(features)
    prediction_class = (prediction > 0.5).astype(int)
    return 'Approved' if prediction_class[0][0] == 1 else 'Rejected'

def make_prediction(model, features):
    prediction = model.predict(features)
    return 'Approved' if prediction[0] == 1 else 'Rejected'

@app.route('/predict_tf', methods=['GET','POST'])
def predict_tf():
    if request.method == 'POST':
        features = get_features(request)
        features_scaled = scaler.transform(features)
        result = make_prediction_tf(model, features_scaled)
        return render_template('result.html', prediction=result)

    return render_template('predict_tf.html')

@app.route('/predict_rf', methods=['GET','POST'])
def predict_rf():
    if request.method == 'POST':
        features = get_features(request)
        result = make_prediction(model_rf, features)
        return render_template('result.html', prediction=result)

    return render_template('predict_rf.html')

@app.route('/predict_XGB', methods=['GET','POST'])
def predict_XGB():
    if request.method == 'POST':
        features = get_features(request)
        result = make_prediction(model_XGB, features)
        return render_template('result.html', prediction=result)

    return render_template('predict_XGB.html')

@app.route('/predict_dt', methods=['GET','POST'])
def predict_dt():
    if request.method == 'POST':
        features = get_features(request)
        result = make_prediction(model_dt, features)
        return render_template('result.html', prediction=result)

    return render_template('predict_dt.html')

if __name__=='__main__':
    app.run(debug=True, port=8080)
