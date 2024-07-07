import streamlit as st

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD # NOTE: this was originally written with an older version of SGD, so I'm using the legacy version just to implement what they had as they had it
import pandas as pd
import pickle
import random
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle

stemmer = LancasterStemmer()
model = keras.saving.load_model("nn_model.keras")
with open ('classes.pickle', 'rb') as file:
    classes = pickle.load(file)
with open ('words.pickle', 'rb') as file:
    words = pickle.load(file)
responses_df = pd.read_csv('responses_dataset_augmented.csv')

nltk.download('punkt')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    return return_list


st.title("Generic Medical Chatbot")
st.caption("A generic medical chatbot created for 5530-0001 Summer 2024 project")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = classify_local(prompt)
    if response[0][0] in responses_df['intent'].values:
        msg = responses_df.responses[responses_df['intent'] == response[0][0]].item()
    else:
        msg = responses_df.responses[responses_df['intent'] == 'noanswer'].item()
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)