
import os
import gradio as gr
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv

load_dotenv()

# ── Global state ─────────────────────────────────────────────────────
vectorstore = None
qa_chain = None

# ── 1. Lazy LLM init (only called when needed) ───────────────────────
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

# ── 2. PDF loading ────────────────────────────────────────────────────
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""
    return raw_text

# ── 3. Build vector store from PDF ───────────────────────────────────
def build_vectorstore(pdf_path):
    text = load_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vs = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./hr_chroma_db"
    )
    return vs

# ── 4. Build RAG chain ────────────────────────────────────────────────
def build_qa_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return chain

# ── 5. Upload handler ─────────────────────────────────────────────────
def upload_pdf(file):
    global vectorstore, qa_chain
    if file is None:
        return "No file uploaded."
    try:
        vectorstore = build_vectorstore(file.name)
        qa_chain = build_qa_chain(vectorstore)
        return "✅ PDF uploaded and indexed. You can now ask questions."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ── 6. Question handler ───────────────────────────────────────────────
def ask_question(question, history):
    if not question.strip():
        return history
    if qa_chain is None:
        history.append((question, "⚠️ Please upload an HR policy PDF first."))
        return history
    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        sources = set(doc.page_content[:100] for doc in result["source_documents"])
        source_text = "\n\n📎 **Sources used:**\n" + "\n".join(f"• {s}…" for s in sources)
        history.append((question, answer + source_text))
    except Exception as e:
        history.append((question, f"❌ Error generating answer: {str(e)}"))
    return history

# ── 7. Gradio UI ──────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #0f1117 !important;
    color: #e8e8e8 !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* Header */
#header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 2rem;
}

#header h1 {
    font-size: 1.8rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem;
}

#header p {
    color: #6b7280;
    font-size: 0.9rem;
    margin: 0;
}

#header .badge {
    display: inline-block;
    background: #1c2333;
    border: 1px solid #2d3748;
    color: #60a5fa;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-top: 0.6rem;
}

/* Upload section */
.upload-row {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

/* File upload */
.gr-file-upload {
    background: #0f1117 !important;
    border: 1px dashed #2d3748 !important;
    border-radius: 8px !important;
    color: #9ca3af !important;
}

/* Status box */
.gr-textbox textarea {
    background: #0f1117 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 8px !important;
    color: #9ca3af !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Chatbot messages */
.gr-chatbot {
    background: #13161f !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
}

.gr-chatbot .message.user {
    background: #1c2a4a !important;
    border-radius: 8px !important;
    color: #bfdbfe !important;
}

.gr-chatbot .message.bot {
    background: #1a1d27 !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
}

/* Input textbox */
.gr-textbox.gr-input textarea {
    background: #13161f !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gr-textbox.gr-input textarea:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* Buttons */
.gr-button {
    background: #1d4ed8 !important;
    border: none !important;
    border-radius: 7px !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: background 0.2s !important;
}

.gr-button:hover {
    background: #2563eb !important;
}

.gr-button.secondary {
    background: #1e2130 !important;
    color: #9ca3af !important;
}
"""

with gr.Blocks(css=CSS, title="HR Policy Chatbot") as app:

    gr.HTML("""
    <div id="header">
        <h1>📋 HR Policy Chatbot</h1>
        <p>Upload your HR policy document and ask questions in plain English</p>
        <span class="badge">Powered by IBM watsonx.ai · granite-13b-chat-v2</span>
    </div>
    """)

    with gr.Row(elem_classes="upload-row"):
        pdf_input = gr.File(
            label="Upload HR Policy PDF",
            file_types=[".pdf"],
            scale=2
        )
        upload_status = gr.Textbox(
            label="Status",
            interactive=False,
            placeholder="Waiting for PDF upload...",
            scale=3
        )

    pdf_input.change(upload_pdf, inputs=pdf_input, outputs=upload_status)

    chatbot = gr.Chatbot(
        label="Conversation",
        height=420,
        bubble_full_width=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            label="",
            placeholder="e.g. What is the leave policy for new employees?",
            scale=5,
            lines=1,
        )
        send_btn = gr.Button("Send", scale=1)

    clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

    msg.submit(ask_question, inputs=[msg, chatbot], outputs=chatbot)
    msg.submit(lambda: "", outputs=msg)
    send_btn.click(ask_question, inputs=[msg, chatbot], outputs=chatbot)
    send_btn.click(lambda: "", outputs=msg)
    clear_btn.click(lambda: [], outputs=chatbot)

    gr.HTML("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem; color: #374151; font-size: 0.78rem;">
        Answers are grounded in your uploaded policy document · Not a substitute for official HR guidance
    </div>
    """)

app.launch(share=False)
