import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="📋",
    layout="centered"
)

st.title("📋 HR Policy Chatbot")
st.caption("Upload your HR policy PDF and ask questions in plain English.")

# ── Session state ─────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Step 1: Upload PDF ────────────────────────────────────────────────
st.sidebar.header("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose an HR Policy PDF", type=["pdf"])

if uploaded_file and uploaded_file.name != st.session_state.doc_name:
    with st.spinner("Reading and indexing your PDF..."):

        # Step 2: Read PDF
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() or ""

        # Step 3: Chunk the document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_text(raw_text)

        # Step 4: Convert to embeddings (free, local)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Step 5: Store in FAISS vector DB
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Step 6: Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Step 7: Build RAG chain
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        st.session_state.doc_name = uploaded_file.name
        st.session_state.chat_history = []

    st.sidebar.success(f"✅ Indexed: {uploaded_file.name}")
    st.sidebar.info(f"📊 {len(reader.pages)} pages · {len(chunks)} chunks")

elif st.session_state.doc_name:
    st.sidebar.success(f"✅ Active: {st.session_state.doc_name}")

# ── Sidebar: clear chat ───────────────────────────────────────────────
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# ── Step 8: Chat UI ───────────────────────────────────────────────────
if not st.session_state.qa_chain:
    st.info("👈 Upload an HR Policy PDF from the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if question := st.chat_input("Ask about your HR policy..."):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain({"query": question})
                answer = result["result"]
                sources = list(set(
                    doc.page_content[:100] for doc in result["source_documents"]
                ))

            st.markdown(answer)

            with st.expander("📎 Source excerpts used"):
                for i, src in enumerate(sources, 1):
                    st.caption(f"{i}. {src}…")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
