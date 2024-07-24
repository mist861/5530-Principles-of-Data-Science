# 5530-0001 Summer 2024 Team Project: Medical Chatbot

This is the Team Project for Team Rocket in 5530 Principles of Big Data Science Section 1. It contains the following directories:

* data_raw: Contains the raw intents.json dataset, the derived raw responses dataset, and the manually-created new intents dataset 
* data_augmented: Contains the combined, augmented intents and the modified responses datasets
* results: Contains the saved model files and their dependencies
* src: Contains the Jupyter notebook with the relevant code and the Python scripts required to run the chatbot
* presentation: Contains the PowerPoint slide deck used in the class presentation

## Execution:

The chatbot can be run by navigating to the src directory and running the following:

* Neural Network (NN):
```
streamlit run nn_model_streamlit.py
```
* Retrieval Augmented Generated (RAG) with Ollama LLM:
```
streamlit run rag_model_streamlit.py
```


## Requirements:

The chatbot can run on any machine capable of running a leightweight LLM (for the RAG) or a NN. The following soft dependencies are required:

* streamlit >= 1.36.0
* streamlit-authenticator >= 0.3.2
* Flask >= 3.0.3
* Flask-Cors >= 4.0.1
* ollama >= 0.2.1
* torch >= 2.3.1
* semantic-router >= 0.0.53
* nltk >= 3.8.1
* keras >= 3.4.1
* pandas: any
* pickle: any
* random: any
* json: any
* yaml: any
* numpy: any

Please note that additional environment variables may need to be set for ollama to run the LLM on GPU. The LLM can be changed as desired by modifying:
```
msg = ollama.chat(model='<model>'...)
```

## Contributors:

Reed White, Manuel Buffa

## Github:

https://github.com/mist861/5530-Principles-of-Data-Science/tree/main/Project
