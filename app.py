from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            age=request.form.get('age'),
            sex=request.form.get('sex'),
            bmi=int(request.form.get('bmi')),
            children=int(request.form.get('children')),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)[0]
        
        return render_template('home.html', result=round(result, 2))
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)