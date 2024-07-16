import streamlit as st
import streamlit_authenticator as stauth

import pandas as pd
import pickle
import random
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import json

import ollama
import torch
from semantic_router import Route
from semantic_router import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

import yaml
from yaml.loader import SafeLoader
with open('streamlit_auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

encoder = HuggingFaceEncoder()
rl = RouteLayer.from_json("../results/rag_model.json")
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

def generate_rag_respose(prompt): # Generate a method to use the RAG to call the LLM
    intent = rl(prompt).name # First, genreate an intent prediction
    if intent != None: # If the predicted intent is not None:
        response = responses_df.responses[responses_df['intent'] == intent].item() # Set the suggested response, from the responses_df, to the suggestion for that intent
    else:
        response = responses_df.responses[responses_df['intent'] == 'noanswer'].item() # Else set it to the suggestion for the noanswer intent
    msg = ollama.chat(model='superdrew100/kappa-3-phi-3-4k-instruct-abliterated', messages = [ 
        {"role": "system", "content": f"You are a helpful medical assistant. Your response should be similar to: {response}"}, 
        {"role": "user", "content": prompt}
    ]) # Invoke the LLM
    return msg['message']['content']

st.title("Generic Medical Chatbot") # Set the page title
st.caption("A generic medical chatbot created for 5530-0001 Summer 2024 project") # Set the page caption

if "messages" not in st.session_state: # If there are no messages in the session state, display a default message
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages: # Display the messages in the session state (right now, just the above)
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(): # If anything is added to the prompt by the user
    st.session_state.messages.append({"role": "user", "content": prompt}) # Add it to the session state
    st.chat_message("user").write(prompt) # Show it in the chat log
    msg = generate_rag_respose(prompt) # Generate a message by calling the above function
    st.session_state.messages.append({"role": "assistant", "content": msg}) # Add the message to the session state
    st.chat_message("assistant").write(msg) # Write the message out in the UI