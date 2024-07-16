import streamlit as st
import streamlit_authenticator as stauth

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle

import yaml
from yaml.loader import SafeLoader
with open('streamlit_auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

nltk.download('punkt')

stemmer = LancasterStemmer()
model = keras.saving.load_model("../results/nn_model.keras")
with open ('../results/intents.pickle', 'rb') as file:
    intents = pickle.load(file)
with open ('../results/words.pickle', 'rb') as file:
    words = pickle.load(file)
responses_df = pd.read_csv('../data_augmented/responses_dataset_augmented.csv')

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
) # Load the streamlit authenticator

authenticator.login() # Initialize the login page

if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

def clean_up_sentence(sentence): # Define a method to preprocess sentences for classification
    sentence_words = nltk.word_tokenize(sentence) # Tokenize the sentence
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words] # Stem each word
    return sentence_words

def bow(sentence, words, show_details=True): # Define a method to turn the sentence into a bag of words, for classification
    sentence_words = clean_up_sentence(sentence) # Tokenize/stem the sentence using the above
    bag = [0]*len(words) # Create a bag of words
    for s in sentence_words: # For each word in the sentence
        for i,w in enumerate(words): # For each word in the original training set
            if w == s: # If the sentence word matches the training word
                bag[i] = 1 # Set its position to 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def classify_local(sentence): # Define a method to classify sentences
    ERROR_THRESHOLD = 0.25 # Set an internal error threshold, any predictions lower than this will not be included in the results
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input']) # Take the input sentence, the original training words, and apply the above functions to them
    results = model.predict([input_data])[0] # Call the NN model to predict the intent
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD] # Return any predictions with a confidence threshold above ERROR_THRESHOLD
    results.sort(key=lambda x: x[1], reverse=True) # Sort by confidence
    return_list = [] # Define an empty array to store results
    for r in results: # For each result:
        return_list.append((intents[r[0]], str(r[1]))) # Append to the return list
    return return_list


st.title("Generic Medical Chatbot") # Set the page title
st.caption("A generic medical chatbot created for 5530-0001 Summer 2024 project") # Set the page caption

if "messages" not in st.session_state: # If there are no messages in the session state, display a default message
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages: # Display the messages in the session state (right now, just the above)
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(): # If anything is added to the prompt by the user
    st.session_state.messages.append({"role": "user", "content": prompt}) # Add it to the session state
    st.chat_message("user").write(prompt) # Show it in the chat log
    response = classify_local(prompt) # Generate a response with the above functions
    if float(response[0][1]) < 0.6: # If the intent has a confidence of less than 60%:
        msg = responses_df.responses[responses_df['intent'] == 'noanswer'].item() # Set it to the noanswer response
    elif response[0][0] in responses_df['intent'].values: # Otherwise, if it's a valid intent:
        msg = responses_df.responses[responses_df['intent'] == response[0][0]].item()  # Set it to the appropriate response
    else: # Otherwise, in a scenario where it goes crazy:
        msg = responses_df.responses[responses_df['intent'] == 'noanswer'].item() # Default to the noanswer response
    st.session_state.messages.append({"role": "assistant", "content": msg}) # Add the message to the session state
    st.chat_message("assistant").write(msg) # Write the message out in the UI