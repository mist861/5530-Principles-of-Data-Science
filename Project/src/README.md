# 5530-0001 Summer 2024 Team Project: Medical Chatbot Source Files

This directory contains the source files used in building the Medical Chatbot:

* 5530-0001_Summer-2024_Project_Chatbot_Team-Rocket_local.ipynb: Jupyter notebook used to load/clean/augment the intent/response datasets and to train/test the NN and RAG intent identifiers
* nn_model_streamlit.py: Python script used to load the Streamlit UI and NN for the NN version of the chatbot
* rag_model_streamlit.py: Python script used to load the Streamlit UI, RAG, and Phi 3 Instruct LLM for the RAG version of the chatbot
* streamlit_auth.yaml: YAML file that defines the basic Streamlit auth configurations

Additionally, the following files are here only for informational purposes. They were used in attempts to finetune/train our own LLM, but ended up not being used in the final product. All datasets used in the training of LLMs are fully separate from the intent identification datasets used above.

* LLM_DataPrep.ipynb: Jupyter notebook that loaded and preprocessed LLM training data
* LLM_TestTrainCloud.ipynb: Jupyter notebook that trained a Phi 1-5 LLM

## Execution:

* Neural Network (NN):

```
streamlit run nn_model_streamlit.py
```

* Retrieval Augmented Generated (RAG) with Ollama Phi 3 Instruct LLM:

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