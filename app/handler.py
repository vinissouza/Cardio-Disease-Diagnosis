import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from cardiodiagnostic.CardioDiagnostic import CardioDiagnostic

# load model
model = pickle.load(open(r'model/model_cardio_catch_diseases_v01.pkl', 'rb'))

# initialize API
app = Flask(__name__)


@app.route('/cardiodiagnostic/predict', methods=['POST'])
def cardiodiagnostic_predict():
    test_json = request.get_json()

    if test_json:  # there is data
        if isinstance(test_json, dict):  # unique example
            test_raw = pd.DataFrame(test_json, index=[0])

        else:  # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # Instantiate Rossmann class
        pipeline = CardioDiagnostic()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        df3 = pipeline.data_preparation(df2)

        # feature selection
        df4 = pipeline.feature_selection(df3)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df4)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)