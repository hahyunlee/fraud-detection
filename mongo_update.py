import pickle
import json
from datetime import datetime
#from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import requests
from pymongo import MongoClient, UpdateOne
import time
from main import *
from sklearn.ensemble import RandomForestClassifier
# from model import
import numpy as np


def trained_model_outputs(model, json_string):
    df0 = pd.DataFrame.from_dict([json_string])
    #df0 = pd.read_json(json.dumps(json_string))
    # df0 = pd.read_json(json_string))
    df = pipeline(df0)
    varz = ['body_length', 'event_length', 'previous_payouts', 'channels',
            'delivery_method', 'name_length', 'user_age', 'user_type']
    test = df[varz]
    probability = model.predict_proba(test)[0][1]
    prediction = model.predict(test)[0]
    return probability, prediction


#app.run(host='0.0.0.0', port=3333, debug=True)
# unpickle
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
hi_threshold = .95
sequence_number = 0
client = MongoClient('mongodb://localhost:27017/')
db = client['testing_fraud']
# db.collection.remove({''})
for x in db.collection.find():
    x["_id"] = 0
    _id = x["object_id"]
# api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
# url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
# response = requests.post(
#     url, json={'api_key': api_key, 'sequence_number': sequence_number})
# raw_data = response.json()

# if raw_data['_next_sequence_number'] != sequence_number:
# sequence_number = raw_data['_next_sequence_number']
    probability, prediction = trained_model_outputs(model, x)
    if prediction == 1:
        _color = 'Yellow'
        if probability > hi_threshold:
            _color = 'Red'
    else:
        _color = 'Green'
#     x['probability'] = probability
#     x['model_label'] = prediction
#     x['prediction'] = _color
    print(probability, prediction, _color)
    print(type(probability), type(prediction), type(_color))
#df = pd.DataFrame.from_dict(raw_data['data'][0],orient = 'index').T

    db.collection.bulk_write([
        UpdateOne({"object_id": _id}, {"$set": {"probability": probability}}),
        #
        UpdateOne({"object_id": _id}, {"$set": {"prediction": _color}})
    ])
