import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="📋",
    layout="centered"
)

st.title("📋 HR Policy Chatbot")
st.caption("Powered by IBM watsonx.ai · Granite LLM · RAG")

# ── Session state ─────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Watson LLM (lazy init) ────────────────────────────────────────────
def get_llm():
    return WatsonxLLM(
        model_id="ibm/granite-13b-chat-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=os.environ.get("WATSONX_PROJECT_ID"),
        apikey=os.environ.get("WATSONX_API_KEY"),
        params={
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.TEMPERATURE: 0.1,
            GenParams.REPETITION_PENALTY: 1.1,
        }
    )

# ── Sidebar: Upload PDF ───────────────────────────────────────────────
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

        # Step 3: Embed (free, local — no extra API key)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Step 4: Store in FAISS
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Step 5: Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Step 6: RAG chain with Watson LLM
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
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

# ── Sidebar: Clear chat ───────────────────────────────────────────────
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("IBM watsonx.ai · granite-13b-chat-v2")

# ── Main: Chat UI ─────────────────────────────────────────────────────
if not st.session_state.qa_chain:
    st.info("👈 Upload an HR Policy PDF from the sidebar to get started.")

else:
    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask about your HR policy..."):

        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer from Watson
        with st.chat_message("assistant"):
            with st.spinner("Asking Watson AI..."):
                try:
                    result = st.session_state.qa_chain({"query": question})
                    answer = result["result"]
                    sources = list(set(
                        doc.page_content[:100] for doc in result["source_documents"]
                    ))
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
