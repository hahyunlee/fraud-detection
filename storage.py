import pickle
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
from pymongo import MongoClient, UpdateOne
import json
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from main import *
import requests
import time
#app = Flask(__name__)
def trained_model_outputs(model, json_string):
    df0 = pd.DataFrame.from_dict([json_string])
    #df0 = pd.read_json(json.dumps(json_string))
    #df0 = pd.read_json(json_string))
    df = pipeline(df0)
    varz = ['body_length','event_length','previous_payouts','channels','delivery_method','name_length','user_age','user_type']
    test = df[varz]
    probability = model.predict_proba(test)[0][1]
    prediction = model.predict(test)[0]
    return probability, prediction



if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=3333, debug=True)
    # unpickle
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    hi_threshold = .95
    sequence_number = 0
    client = MongoClient('mongodb://localhost:27017/')
    db = client['testing_fraud']
    api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
    url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
    while True:
        response = requests.post(
            url, json={'api_key': api_key, 'sequence_number': sequence_number})
        raw_data = response.json()
        if raw_data['_next_sequence_number'] != sequence_number:
            sequence_number = raw_data['_next_sequence_number']
            probability, prediction = trained_model_outputs(
                model,raw_data['data'][0])
		
            if prediction == 1:
                raw_data['data'][0]['prediction'] = 'Yellow'
                if probability > hi_threshold:
                    raw_data['data'][0]['prediction'] = 'Red'
            else:
                raw_data['data'][0]['prediction'] = 'Green'
            raw_data['data'][0]['probability'] = probability
            #df = pd.DataFrame.from_dict(raw_data['data'][0],orient = 'index').T
            db.collection.insert_one(raw_data['data'][0])
        #print(sequence_number, raw_data['data'][0].keys())
        time.sleep(120)
