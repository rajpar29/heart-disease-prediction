import json

import requests
import csv

import tensorflow as tf
import pandas as pd
import numpy as np


def convert_num(string):
    # print(string.split('.'))
    if len(string.split('.')) > 1:
        return float(string)
    else:
        # print("int",int(string))
        return int(string)


data_list = []
target_list = []
csv_columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
with open('data/heart.csv') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        temp_dict = {}
        temp_dict['age'] = convert_num(row[0])
        temp_dict['sex'] = convert_num(row[1])
        temp_dict['cp'] = convert_num(row[2])
        temp_dict['trestbps'] = convert_num(row[3])
        temp_dict['chol'] = convert_num(row[4])
        temp_dict['fbs'] = convert_num(row[5])
        temp_dict['restecg'] = convert_num(row[6])
        temp_dict['thalach'] = convert_num(row[7])
        temp_dict['exang'] = convert_num(row[8])
        temp_dict['oldpeak'] = convert_num(row[9])
        temp_dict['slope'] = convert_num(row[10])
        temp_dict['ca'] = convert_num(row[11])
        temp_dict['thal'] = convert_num(row[12])
        target_list.append(row[13])
        data_list.append(temp_dict)
    # data_list.pop(0)

index = 0
for data,target in zip(data_list, target_list):
    req = requests.post("http://localhost:5000/predict",json=data)
    print("==================", index, "===================")
    print(data)
    print(json.loads(req.content),target)
    print('\n\n')
    index = index +1
    # break

