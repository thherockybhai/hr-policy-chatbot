import os
import requests
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ── Watson REST API (no ibm-watsonx-ai SDK needed) ────────────────────
WATSON_URL = "https://us-south.ml.cloud.ibm.com"
IAM_URL    = "https://iam.cloud.ibm.com/identity/token"

def get_iam_token(api_key: str) -> str:
    """Exchange IBM API key for a bearer token."""
    resp = requests.post(
        IAM_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def ask_watson(question: str, context: str) -> str:
    """Send question + retrieved context to Watson Granite via REST."""
    api_key    = os.environ.get("WATSONX_API_KEY", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")

    if not api_key or not project_id:
        return "❌ WATSONX_API_KEY or WATSONX_PROJECT_ID not set in secrets."

    token = get_iam_token(api_key)

    prompt = f"""You are an HR policy assistant. Answer the employee's question using only the HR policy excerpts provided below.
If the answer is not in the excerpts, say "I could not find this in the HR policy document."

HR Policy Excerpts:
{context}

Employee Question: {question}

Answer:"""

    payload = {
        "model_id": "ibm/granite-13b-chat-v2",
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 512,
            "repetition_penalty": 1.1,
            "temperature": 0.1,
        },
        "project_id": project_id,
    }

    resp = requests.post(
        f"{WATSON_URL}/ml/v1/text/generation?version=2023-05-29",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["results"][0]["generated_text"].strip()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="📋",
    layout="centered"
)

st.title("📋 HR Policy Chatbot")
st.caption("Powered by IBM watsonx.ai · Granite LLM · RAG")

# ── Session state ─────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose an HR Policy PDF", type=["pdf"])

if uploaded_file and uploaded_file.name != st.session_state.doc_name:
    with st.spinner("Reading and indexing your PDF..."):

        # Step 1: Read PDF
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() or ""

        # Step 2: Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_text(raw_text)

        # Step 3: Embed locally (free, no API key)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Step 4: FAISS vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        st.session_state.doc_name = uploaded_file.name
        st.session_state.chat_history = []

    st.sidebar.success(f"✅ Indexed: {uploaded_file.name}")
    st.sidebar.info(f"📊 {len(reader.pages)} pages · {len(chunks)} chunks")

elif st.session_state.doc_name:
    st.sidebar.success(f"✅ Active: {st.session_state.doc_name}")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("IBM watsonx.ai · granite-13b-chat-v2")

# ── Chat UI ───────────────────────────────────────────────────────────
if not st.session_state.retriever:
    st.info("👈 Upload an HR Policy PDF from the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask about your HR policy..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Asking Watson AI..."):
                try:
                    # Retrieve relevant chunks
                    docs = st.session_state.retriever.invoke(question)
                    context = "\n\n".join(doc.page_content for doc in docs)
                    sources = [doc.page_content[:100] for doc in docs]

                    # Call Watson via REST
                    answer = ask_watson(question, context)

                    st.markdown(answer)
                    with st.expander("📎 Source excerpts used"):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"{i}. {src}…")

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    err = f"❌ Error: {str(e)}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err}
                    )
