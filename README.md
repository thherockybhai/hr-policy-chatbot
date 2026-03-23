
# 📋 HR Policy Chatbot

A RAG-powered HR policy chatbot built with **IBM watsonx.ai** (Granite LLM), **LangChain**, **ChromaDB**, and **Gradio**.

Upload any HR policy PDF and ask questions in plain English — answers are grounded in your document with source references.

---

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                            ↓
User Question → Query Embedding → Similarity Search → Top-K Chunks
                                                            ↓
                                              Watson Granite LLM → Answer
```

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | IBM watsonx.ai `granite-13b-chat-v2` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector DB | ChromaDB (persisted to disk) |
| Orchestration | LangChain `RetrievalQA` |
| UI | Gradio |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/hr-policy-chatbot.git
cd hr-policy-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env` and fill in your IBM watsonx credentials:
```
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
```

> Get your API key from [IBM Cloud](https://cloud.ibm.com/iam/apikeys)
> Get your Project ID from [watsonx.ai](https://dataplatform.cloud.ibm.com)

### 5. Run the app
```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## Usage

1. Upload your HR policy PDF using the file uploader
2. Wait for the "✅ PDF uploaded and indexed" confirmation
3. Type your question in plain English and press Send
4. The bot answers using only your document, with source excerpts shown

---

## Deployment

### Option A — Hugging Face Spaces (free, recommended)
See [deployment guide](https://huggingface.co/docs/hub/spaces-sdks-gradio) — push this repo with a `README.md` that includes `sdk: gradio`.

### Option B — Railway / Render
Set the environment variables in the platform dashboard and point the start command to `python app.py`.

---

## License
MIT
