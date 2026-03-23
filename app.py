import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="HR Policy Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────
qa_chain = None
doc_meta = {"name": None, "pages": 0, "chunks": 0}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────
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

# ── PDF processing ────────────────────────────────────────────────────
def process_pdf(pdf_path: str, filename: str):
    global qa_chain, doc_meta

    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if Path("./hr_chroma_db").exists():
        shutil.rmtree("./hr_chroma_db")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./hr_chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    doc_meta = {
        "name": filename,
        "pages": len(reader.pages),
        "chunks": len(chunks)
    }

    return doc_meta

# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        meta = process_pdf(str(save_path), file.filename)
        return {"success": True, "meta": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(body: QuestionRequest):
    if qa_chain is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = qa_chain({"query": body.question})
        answer = result["result"]
        sources = list(set(
            doc.page_content[:120] for doc in result["source_documents"]
        ))
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    return {
        "ready": qa_chain is not None,
        "document": doc_meta
    }

# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
