import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from ibm_watsonx_ai.foundation_models import Model

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="📋",
    layout="centered"
)

st.title("📋 HR Policy Chatbot")
st.caption("Powered by IBM watsonx.ai · Granite · RAG")

# ── Session state ─────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "model" not in st.session_state:
    st.session_state.model = None

# ── Watson Model ──────────────────────────────────────────────────────
def get_model():
    return Model(
        model_id="ibm/granite-3-8b-instruct",
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 300,
        },
        credentials={
            "apikey": os.environ.get("WATSONX_API_KEY"),
            "url": "https://eu-de.ml.cloud.ibm.com",
        },
        project_id=os.environ.get("WATSONX_PROJECT_ID"),
    )

# ── File loaders per format ───────────────────────────────────────────
def load_file(uploaded_file) -> list[Document]:
    ext = uploaded_file.name.split(".")[-1].lower()

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

        elif ext == "docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()

        elif ext == "txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()

        elif ext in ("xlsx", "xls"):
            import pandas as pd
            if ext == "xlsx":
                df = pd.read_excel(tmp_path, engine="openpyxl")
            else:
                df = pd.read_excel(tmp_path, engine="xlrd")

            # Convert each row to a document
            docs = []
            for i, row in df.iterrows():
                content = " | ".join(
                    f"{col}: {val}"
                    for col, val in row.items()
                    if str(val).strip() not in ("", "nan", "None")
                )
                if content.strip():
                    docs.append(Document(
                        page_content=content,
                        metadata={"row": i, "source": uploaded_file.name}
                    ))

        elif ext == "csv":
            import pandas as pd
            df = pd.read_csv(tmp_path)
            docs = []
            for i, row in df.iterrows():
                content = " | ".join(
                    f"{col}: {val}"
                    for col, val in row.items()
                    if str(val).strip() not in ("", "nan", "None")
                )
                if content.strip():
                    docs.append(Document(
                        page_content=content,
                        metadata={"row": i, "source": uploaded_file.name}
                    ))

        else:
            raise ValueError(f"Unsupported file type: .{ext}")

        return docs

    finally:
        os.unlink(tmp_path)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("📄 Upload Document")
st.sidebar.caption("Supports PDF, DOCX, TXT, XLSX, XLS, CSV")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "txt", "xlsx", "xls", "csv"]
)

if uploaded_file and uploaded_file.name != st.session_state.doc_name:
    with st.spinner(f"Reading and indexing {uploaded_file.name}..."):
        try:
            # Step 1: Load file
            documents = load_file(uploaded_file)

            if not documents:
                st.sidebar.error("❌ No content found in file.")
            else:
                # Step 2: Chunk
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                )
                chunks = splitter.split_documents(documents)

                # Step 3: Embed locally
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Step 4: FAISS vector store
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []

                # Step 5: Init Watson model once
                st.session_state.model = get_model()

                st.sidebar.success(f"✅ Indexed: {uploaded_file.name}")
                st.sidebar.info(f"📊 {len(documents)} sections · {len(chunks)} chunks")

        except Exception as e:
            st.sidebar.error(f"❌ Failed to load file: {str(e)}")

elif st.session_state.doc_name:
    st.sidebar.success(f"✅ Active: {st.session_state.doc_name}")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("IBM watsonx.ai · granite-3-8b-instruct · eu-de")

# ── Chat UI ───────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.info("👈 Upload an HR Policy document from the sidebar to get started.")
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
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    docs = retriever.invoke(question)
                    context = "\n\n".join(doc.page_content for doc in docs)
                    sources = [doc.page_content[:100] for doc in docs]

                    prompt = f"""You are an HR assistant. Answer ONLY from the context below.
If the answer is not in the context, say: I could not find this in the HR policy document.

Context:
{context}

Question:
{question}

Answer:"""

                    answer = st.session_state.model.generate_text(prompt=prompt)

                    st.markdown(answer)
                    with st.expander("📎 Source excerpts used"):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"{i}. {src}…")

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    err = f"❌ Watson AI error: {str(e)}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err}
                    )
