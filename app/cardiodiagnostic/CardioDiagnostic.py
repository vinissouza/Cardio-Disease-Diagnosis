import pickle
import numpy as np


class CardioDiagnostic():
    def __init__(self):
        #self.home_path = r''
        self.home_path = r'C:\Users\Ganso\Documents\Data-Science\Gitlab\pa001_cardio_catch_diseases\webapp'
        self.age_scaler = pickle.load(open(self.home_path + '/parameter/age_scaler.pkl', 'rb'))
        self.height_scaler = pickle.load(open(self.home_path + '/parameter/height_scaler.pkl', 'rb'))
        self.weight_scaler = pickle.load(open(self.home_path +  '/parameter/weight_scaler.pkl', 'rb'))
        self.bmi_scaler = pickle.load(open(self.home_path + '/parameter/bmi_scaler.pkl', 'rb'))
        self.ap_hi_scaler = pickle.load(open(self.home_path + '/parameter/ap_hi_scaler.pkl', 'rb'))
        self.ap_lo_scaler = pickle.load(open(self.home_path + '/parameter/ap_lo_scaler.pkl', 'rb'))
        self.ap_ratio_scaler = pickle.load(open(self.home_path + '/parameter/ap_ratio_scaler.pkl', 'rb'))
        self.ap_stages_encoder = pickle.load(open(self.home_path + '/parameter/ap_stages_encoder.pkl', 'rb'))
        self.weight_status_encoder = pickle.load(open(self.home_path + '/parameter/weight_status_encoder.pkl', 'rb'))
        self.cholesterol_encoder = pickle.load(open(self.home_path + '/parameter/cholesterol_encoder.pkl', 'rb'))
        self.gluc_encoder = pickle.load(open(self.home_path + '/parameter/gluc_encoder.pkl', 'rb'))
        self.feat_imp_list = pickle.load(open('../parameter/feat_select_imp.pkl', 'rb'))

    def data_cleaning(self, data):
        # convert age to years
        data['age'] = np.round(data['age']/365, 2)

        # convert gender to binary
        data['gender'] = data['gender'] - 1

        # drop id column
        data = data.drop(['id'], axis=1)

        return data

    def feature_engineering(self, data):
        # create a feature for body mass index (BMI)
        data['bmi'] = np.round(data['weight']/((data['height']/100)**2), 2)

        # create a feature for (ap_hi / api_lo)
        data['ap_hi/ap_lo'] = data[['ap_hi', 'ap_lo']].apply(lambda x: np.round(x['ap_hi']/x['ap_lo'], 2)
                                                                       if x['ap_lo'] != 0 else 0, axis=1 )

        # create feature for people who smoke, drink and are not active
        data['bad_habits'] = data[['smoke', 'alco', 'active']].apply(lambda x: 1 if (x['smoke'] == 1) &
                                                                                    (x['alco'] == 1) &
                                                                                    (x['active'] == 0) else 0,
                                                                     axis=1)

        # create feature for weight status based on BMI
        data['weight_status'] = data['bmi'].apply(lambda x: 'underweight' if x < 18.5 else
                                                            'healthy' if (x >= 18.5) & (x < 25) else
                                                            'overweight' if (x >= 25) & (x < 30) else
                                                            'obesity')

        # create a feature for blood pressure stages
        data['ap_stages'] = data[['ap_hi', 'ap_lo']].apply(
                                                lambda x: 'hp_crisis' if (x['ap_hi'] > 180) | (x['ap_lo'] > 120) else
                                                          'hp_stage_2' if (x['ap_hi'] >= 140) | (x['ap_lo'] >= 90) else
                                                          'hp_stage_1' if (x['ap_hi'] >= 130) | (x['ap_lo'] >= 80) else
                                                          'elevated' if (x['ap_hi'] >= 120) else
                                                          'normal',
                                                axis=1)

        # transformation cholesterol in categorical data
        data['cholesterol'] = data['cholesterol'].apply(lambda x: 'normal' if x == 1 else
                                                                  'above_normal' if x == 2 else
                                                                  'well_above_normal')

        # transformation glucose in categorical data
        data['gluc'] = data['gluc'].apply(lambda x: 'normal' if x == 1 else
                                                    'above_normal' if x == 2 else
                                                    'well_above_normal')

        return data

    def data_preparation(self, data):
        # Rescale
        # age
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # height
        data['height'] = self.height_scaler.transform(data[['height']].values)

        # weight
        data['weight'] = self.weight_scaler.transform(data[['weight']].values)

        # bmi
        data['bmi'] = self.bmi_scaler.transform(data[['bmi']].values)

        # ap_hi
        data['ap_hi'] = self.ap_hi_scaler.transform(data[['ap_hi']].values)

        # ap_lo
        data['ap_lo'] = self.ap_lo_scaler.transform(data[['ap_lo']].values)

        # ap_hi/ap_lo
        data['ap_hi/ap_lo'] = self.ap_ratio_scaler.transform(data[['ap_hi/ap_lo']].values)

        # Encoding
        # blood pressure category
        data['ap_stages'] = self.ap_stages_encoder.transform(data['ap_stages'])

        # BMI class
        data['weight_status'] = self.weight_status_encoder.transform(data['weight_status'])

        # cholesterol
        data['cholesterol'] = self.cholesterol_encoder.transform(data['cholesterol'])

        # glucose
        data['gluc'] = self.gluc_encoder.transform(data['gluc'])

        return data

    def feature_selection(self, data):
        # feature selection by importance
        data = data[self.feat_imp_list]

        return data

    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)

        # join pred into the original data
        original_data['prediction'] = pred

        return original_data.to_json(orient='records', date_format='iso')