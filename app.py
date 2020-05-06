from flask import Flask, request
from flask_cors import CORS
from joblib import load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
# Load models

knn_classifier = load('KNN_classifier.joblib.z')
decision_tree_classifier = load('decisionTree.jobliz.z')
random_forest_classifier = load('RandomForestClassifier.joblib.z')
svc_linear_classifier = load('SVC_linear_classifier.jobliz.z')
tf_model = tf.keras.models.load_model('tf_model.h5')
tf_model.build(input_shape=(None, 30))

loaded_clean = pd.read_pickle('clean_data')


def preprocess_input_data(data_frame):
    column_order = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_0', 'sex_1',
                    'cp_0', 'cp_1', 'cp_2', 'cp_3', 'fbs_0', 'fbs_1', 'restecg_0',
                    'restecg_1', 'restecg_2', 'exang_0', 'exang_1', 'slope_0', 'slope_1',
                    'slope_2', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_0', 'thal_1',
                    'thal_2', 'thal_3']

    data_frame = pd.get_dummies(data_frame, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    # print("After creating Dummies  ==>> ", data_frame)
    add_missing_columns(data_frame)
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data_scaler = load('data_scaler')
    data_frame[columns_to_scale] = data_scaler.transform(data_frame[columns_to_scale])
    data_frame = data_frame[column_order]
    # print("After Scaling ==>> ",data_frame.iloc[0],"\n====================\n")
    return data_frame


def add_missing_columns(data_frame):
    orignal_dataset = pd.read_pickle('clean_data')
    # print(orignal_dataset.columns)
    missing_columns = set(orignal_dataset.columns) - set(data_frame.columns)
    for missing_column in missing_columns:
        if missing_column == 'target':
            continue
        data_frame[missing_column] = 0


def predict(data):
    data_frame = pd.DataFrame.from_records([data])
    data_frame = preprocess_input_data(data_frame)
    print(data_frame.values)
    res = tf_model.predict(data_frame.values)[0][0]
    # print(res)
    return {
        'knn_prediction': int(knn_classifier.predict(data_frame)[0]),
        # 'decision_tree': int(decision_tree_classifier.predict(data_frame)[0]),
        # 'random_forest': int(random_forest_classifier.predict(data_frame)[0]),
        # 'svc_linear': int(svc_linear_classifier.predict(data_frame)[0])
        'tensorflow': int(round(res))
    }


@app.route('/predict', methods=['POST'])
def index():
    heart_data = json.loads(request.data)
    # print(heart_data)
    return predict(heart_data)




# print(predi(302))
if __name__ == '__main__':
    app.run()
