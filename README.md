# 📋 HR Policy Chatbot

A RAG-powered HR Policy chatbot built with **IBM watsonx.ai** (Granite LLM), **FAISS**, and **Streamlit**.

Upload any HR policy PDF and ask questions in plain English — answers are grounded in your document with source references shown.

## How it works

```
PDF → chunks → embeddings → FAISS vector DB
Question → embedding → similarity search → top 4 chunks → Watson Granite LLM → answer
```

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | IBM watsonx.ai `granite-13b-chat-v2` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| Vector DB | FAISS (in-memory) |
| Framework | LangChain |
| UI | Streamlit |

## Deploy on Streamlit Cloud (free, no local setup needed)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Click **New app** → connect your repo → set `app.py` as main file
4. Go to **Advanced settings → Secrets** and paste:
   ```toml
   WATSONX_API_KEY = "your_ibm_cloud_api_key"
   WATSONX_PROJECT_ID = "your_watsonx_project_id"
   ```
5. Click **Deploy** — live in ~2 minutes

## Get your Watson credentials

### WATSONX_API_KEY
1. Go to https://cloud.ibm.com/iam/apikeys
2. Click **Create an IBM Cloud API key**
3. Copy the key (shown only once)

### WATSONX_PROJECT_ID
1. Go to https://dataplatform.cloud.ibm.com
2. Create a new project → **Manage → General**
3. Copy the **Project ID**

## File structure

```
hr-policy-chatbot/
├── app.py            ← entire app
├── requirements.txt
├── .env.example      ← copy to .env for local use
├── .gitignore
└── README.md
```
