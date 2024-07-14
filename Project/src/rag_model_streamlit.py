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
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_router import Route
from semantic_router import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

import yaml
from yaml.loader import SafeLoader
with open('streamlit_auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

encoder = HuggingFaceEncoder()
rl = RouteLayer.from_json("rag_model.json")
responses_df = pd.read_csv('responses_dataset_augmented.csv')

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

def generate_rag_respose(prompt):
    intent = rl(prompt).name
    if intent != None:
        response = responses_df.responses[responses_df['intent'] == intent].item()
    else:
        response = responses_df.responses[responses_df['intent'] == 'noanswer'].item()
    msg = ollama.chat(model='superdrew100/kappa-3-phi-3-4k-instruct-abliterated', messages = [ 
        {"role": "system", "content": f"You are a helpful medical assistant. Your response should be similar to: {response}"}, 
        {"role": "user", "content": prompt}
    ])
    return msg['message']['content']

st.title("Generic Medical Chatbot")
st.caption("A generic medical chatbot created for 5530-0001 Summer 2024 project")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = generate_rag_respose(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)