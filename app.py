import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
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
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
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

# ── File loader per format ────────────────────────────────────────────
def load_file(uploaded_file) -> list[Document]:
    ext = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            from langchain_community.document_loaders import PyPDFLoader
            docs = PyPDFLoader(tmp_path).load()

        elif ext == "docx":
            from langchain_community.document_loaders import Docx2txtLoader
            docs = Docx2txtLoader(tmp_path).load()

        elif ext == "txt":
            from langchain_community.document_loaders import TextLoader
            docs = TextLoader(tmp_path, encoding="utf-8").load()

        elif ext in ("xlsx", "xls"):
            import pandas as pd
            engine = "openpyxl" if ext == "xlsx" else "xlrd"
            df = pd.read_excel(tmp_path, engine=engine)
            docs = [
                Document(
                    page_content=" | ".join(
                        f"{col}: {val}" for col, val in row.items()
                        if str(val).strip() not in ("", "nan", "None")
                    ),
                    metadata={"row": i, "source": uploaded_file.name}
                )
                for i, row in df.iterrows()
            ]
            docs = [d for d in docs if d.page_content.strip()]

        elif ext == "csv":
            import pandas as pd
            df = pd.read_csv(tmp_path)
            docs = [
                Document(
                    page_content=" | ".join(
                        f"{col}: {val}" for col, val in row.items()
                        if str(val).strip() not in ("", "nan", "None")
                    ),
                    metadata={"row": i, "source": uploaded_file.name}
                )
                for i, row in df.iterrows()
            ]
            docs = [d for d in docs if d.page_content.strip()]

        else:
            raise ValueError(f"Unsupported file type: .{ext}")

        # Tag every doc with its source filename
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        return docs

    finally:
        os.unlink(tmp_path)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("📄 Upload Documents")
st.sidebar.caption("PDF, DOCX, TXT, XLSX, XLS, CSV · Multiple files allowed")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt", "xlsx", "xls", "csv"],
    accept_multiple_files=True        # ← key change
)

# Index button — only shown when files are uploaded
if uploaded_files:
    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state.indexed_files
    ]

    if new_files:
        if st.sidebar.button(f"📥 Index {len(new_files)} new file(s)"):
            all_chunks = []
            failed = []

            progress = st.sidebar.progress(0, text="Starting...")

            for i, file in enumerate(new_files):
                try:
                    progress.progress(
                        int((i / len(new_files)) * 100),
                        text=f"Loading {file.name}..."
                    )
                    docs = load_file(file)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50,
                    )
                    chunks = splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    st.session_state.indexed_files.append(file.name)

                except Exception as e:
                    failed.append(f"{file.name}: {str(e)}")

            if all_chunks:
                progress.progress(90, text="Building vector index...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                if st.session_state.vectorstore is None:
                    # First time — create fresh
                    st.session_state.vectorstore = FAISS.from_documents(
                        all_chunks, embeddings
                    )
                else:
                    # Add to existing index
                    new_vs = FAISS.from_documents(all_chunks, embeddings)
                    st.session_state.vectorstore.merge_from(new_vs)

                if st.session_state.model is None:
                    st.session_state.model = get_model()

                progress.progress(100, text="Done!")
                st.sidebar.success(f"✅ Indexed {len(new_files)} file(s)")

            if failed:
                for f in failed:
                    st.sidebar.error(f"❌ {f}")
    else:
        st.sidebar.success("✅ All uploaded files already indexed")

# ── Indexed files list ────────────────────────────────────────────────
if st.session_state.indexed_files:
    st.sidebar.divider()
    st.sidebar.markdown("**Indexed documents:**")
    for fname in st.session_state.indexed_files:
        st.sidebar.caption(f"📄 {fname}")

    if st.sidebar.button("🗑️ Clear all documents"):
        st.session_state.vectorstore = None
        st.session_state.indexed_files = []
        st.session_state.chat_history = []
        st.session_state.model = None
        st.rerun()

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("IBM watsonx.ai · granite-3-8b-instruct · eu-de")

# ── Chat UI ───────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.info("👈 Upload one or more documents and click **Index** to get started.")
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
                        search_kwargs={"k": 4}
                    )
                    docs = retriever.invoke(question)
                    context = "\n\n".join(doc.page_content for doc in docs)

                    # Show which documents the answer came from
                    sources = list(set(
                        doc.metadata.get("source", "unknown") for doc in docs
                    ))
                    excerpts = [doc.page_content[:100] for doc in docs]

                    prompt = f"""You are an HR assistant. Answer ONLY from the context below.
If the answer is not in the context, say: I could not find this in the HR policy documents.

Context:
{context}

Question:
{question}

Answer:"""

                    answer = st.session_state.model.generate_text(prompt=prompt)

                    st.markdown(answer)

                    with st.expander("📎 Sources used"):
                        st.caption(f"**Documents:** {', '.join(sources)}")
                        for i, exc in enumerate(excerpts, 1):
                            st.caption(f"{i}. {exc}…")

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    err = f"❌ Watson AI error: {str(e)}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err}
                    )
