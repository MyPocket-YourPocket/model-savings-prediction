import pickle
import streamlit as st
import numpy as np
import json
from json import JSONEncoder

import tensorflow as tf

st.write(st.__version__)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    

scaler_weights_list = []
with open('save_model/scaler_weights.json', 'r') as json_file:
    loaded_weights_json = json.load(json_file)
    
    for i in range(len(loaded_weights_json.keys())):
        if i == 2:
            scaler_weights_list.append(loaded_weights_json[str(i)][0])
        else:
            scaler_weights_list.append(np.array(loaded_weights_json[str(i)]))


# Load the model architecture from JSON file
with open('save_model/mlp.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = tf.keras.models.model_from_json(loaded_model_json)
 
 
# Output confirmation message
print("Model architecture loaded successfully.")

        
loaded_weights_list = []
with open('save_model/mlp_weights.json', 'r') as json_file:
    loaded_weights_json = json.load(json_file)
    
    for i in range(len(loaded_weights_json.keys())):
        loaded_weights_list.append(np.array(loaded_weights_json[str(i)]))

scaler = tf.keras.layers.Normalization(axis=-1, name="Normalization", mean=scaler_weights_list[0], variance=scaler_weights_list[1])
model.set_weights(loaded_weights_list)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    monthly_income = st.text_input(label="Pemasukan Bulanan")
with col2:
    monthly_expenses = st.text_input(label="Pengeluaran Bulanan")
with col3:
    savings_goal = st.text_input(label="Harapan Menabung")


if monthly_income is not None and monthly_expenses is not None and savings_goal is not None :
    predict_button = st.button("Predict", use_container_width=True)
    
    if predict_button:
        input_x = np.array([[float(monthly_income), float(monthly_expenses), float(savings_goal)]])
        input_x = scaler(input_x)
        y_pred1, y_pred2 = model.predict(input_x)
        
        y_pred1 = round(y_pred1[0,0], ndigits=2)
        y_pred2 = round(y_pred2[0,0])
        
        st.write(f"The Prediction of Monthly Saving is **{y_pred1}**")
        st.write(f"The Prediction of Total Saving Installment is **{y_pred2}**")
        
            