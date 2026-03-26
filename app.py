import os
import re
import tempfile
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ibm_watsonx_ai.foundation_models import Model

load_dotenv()

st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── IBM Dark Theme CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* ── Page background ── */
.stApp {
    background-color: #0a0d14 !important;
}

/* ── Hide default Streamlit header/footer ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Custom top header bar ── */
.custom-header {
    background: #0d111c;
    border-bottom: 1px solid #1a2035;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 1.5rem -1rem;
    border-radius: 0;
}
.header-left { display: flex; align-items: center; gap: 10px; }
.logo-mark {
    width: 28px; height: 28px;
    background: #1a56db;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: white;
}
.header-title { font-size: 15px; font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em; }
.header-badge {
    background: #0f2040;
    border: 1px solid #1a3a6e;
    color: #60a5fa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 4px;
}
.dot-green { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #22c55e; margin-right: 5px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d111c !important;
    border-right: 1px solid #1a2035 !important;
}
[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stFileUploader {
    background: #111827 !important;
    border: 1.5px dashed #1e2d45 !important;
    border-radius: 8px !important;
    padding: 8px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1a56db !important;
    color: white !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2563eb !important;
}

/* ── Weather card in sidebar ── */
.weather-card {
    background: #111827;
    border: 1px solid #1a2035;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.weather-city { font-size: 12px; font-weight: 600; color: #60a5fa !important; }
.weather-temp { font-size: 22px; font-weight: 600; color: #e2e8f0 !important; }
.weather-desc { font-size: 11px; color: #64748b !important; }
.weather-row { display: flex; gap: 12px; margin-top: 4px; font-size: 10px; color: #374151 !important; font-family: 'IBM Plex Mono', monospace; }

/* ── Indexed file pills ── */
.file-pill {
    background: #111827;
    border: 1px solid #1a2035;
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 11px;
    color: #64748b;
    margin-bottom: 4px;
    display: block;
}

/* ── Main chat area ── */
.main-chat {
    background: #0a0d14;
    max-width: 780px;
    margin: 0 auto;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessageContent"] {
    background: #111827 !important;
    border: 1px solid #1a2035 !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
}
[data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"] {
    background: #0f2040 !important;
    border-color: #1a3a6e !important;
    color: #bfdbfe !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0d111c !important;
    border-top: 1px solid #1a2035 !important;
    padding: 12px 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #1a56db !important;
}
[data-testid="stChatInput"] button {
    background: #1a56db !important;
    border-radius: 8px !important;
}

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    background: #0d111c !important;
    border: 1px solid #1a2035 !important;
    border-radius: 7px !important;
}
[data-testid="stExpander"] summary {
    color: #374151 !important;
    font-size: 11px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Info/success/error boxes ── */
[data-testid="stAlert"] {
    background: #111827 !important;
    border: 1px solid #1a2035 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: #1a56db !important;
}

/* ── Divider ── */
hr { border-color: #1a2035 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1a2035; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Custom header ─────────────────────────────────────────────────────
st.markdown("""
<div class="custom-header">
    <div class="header-left">
        <div class="logo-mark">W</div>
        <span class="header-title">HR Policy Chatbot</span>
    </div>
    <span class="header-badge"><span class="dot-green"></span>watsonx.ai · granite-3-8b-instruct</span>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
for key, default in {
    "vectorstore": None,
    "chat_history": [],
    "indexed_files": [],
    "model": None,
    "location": None,
    "weather": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Watson Model ──────────────────────────────────────────────────────
def get_model():
    return Model(
        model_id="ibm/granite-3-8b-instruct",
        params={"decoding_method": "greedy", "max_new_tokens": 300},
        credentials={
            "apikey": os.environ.get("WATSONX_API_KEY"),
            "url": "https://eu-de.ml.cloud.ibm.com",
        },
        project_id=os.environ.get("WATSONX_PROJECT_ID"),
    )

# ── Location: resolve city name → lat/lon via Open-Meteo geocoding ───
def resolve_city(city_name: str) -> dict:
    """Convert a city name to lat/lon using Open-Meteo geocoding (free, no key)."""
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=5
        )
        results = r.json().get("results", [])
        if results:
            res = results[0]
            return {
                "city": res.get("name", city_name),
                "region": res.get("admin1", ""),
                "country": res.get("country", ""),
                "lat": res.get("latitude", 12.97),
                "lon": res.get("longitude", 77.59),
            }
    except Exception:
        pass
    # Default to Bengaluru if lookup fails
    return {"city": city_name, "region": "", "country": "", "lat": 12.97, "lon": 77.59}

def get_weather(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weathercode,windspeed_10m,relativehumidity_2m"
            f"&timezone=auto"
        )
        r = requests.get(url, timeout=5)
        data = r.json()
        current = data.get("current", {})
        code = current.get("weathercode", 0)
        temp = current.get("temperature_2m", 25)
        humidity = current.get("relativehumidity_2m", 50)
        wind = current.get("windspeed_10m", 0)
        if code == 0: desc, icon = "Clear skies", "☀️"
        elif code in (1, 2): desc, icon = "Partly cloudy", "⛅"
        elif code == 3: desc, icon = "Overcast", "☁️"
        elif code in (45, 48): desc, icon = "Foggy", "🌫️"
        elif code in (51, 53, 55, 61, 63, 65): desc, icon = "Rainy", "🌧️"
        elif code in (71, 73, 75): desc, icon = "Snowy", "❄️"
        elif code in (95, 96, 99): desc, icon = "Stormy", "⛈️"
        else: desc, icon = "Pleasant", "🌤️"
        return {"temp": round(temp), "desc": desc.lower(), "icon": icon, "humidity": int(humidity), "wind": round(wind), "code": code}
    except Exception:
        pass
    return {"temp": 28, "desc": "pleasant", "icon": "🌤️", "humidity": 60, "wind": 10, "code": 0}

def get_season(lat):
    month = datetime.now().month
    is_northern = lat >= 0
    if is_northern:
        if month in (12, 1, 2): return "winter"
        elif month in (3, 4, 5): return "spring"
        elif month in (6, 7, 8): return "summer"
        else: return "autumn"
    else:
        if month in (12, 1, 2): return "summer"
        elif month in (3, 4, 5): return "autumn"
        elif month in (6, 7, 8): return "winter"
        else: return "spring"

CITY_FOODS = {
    "bengaluru": ["Bisi Bele Bath", "Masala Dosa", "Dum Biryani", "Filter Coffee", "Mysore Pak"],
    "bangalore": ["Bisi Bele Bath", "Masala Dosa", "Dum Biryani", "Filter Coffee", "Mysore Pak"],
    "mumbai": ["Vada Pav", "Pav Bhaji", "Bombay Sandwich", "Cutting Chai"],
    "delhi": ["Butter Chicken", "Chole Bhature", "Paranthe", "Lassi"],
    "chennai": ["Idli Sambar", "Chettinad Chicken", "Filter Coffee", "Pongal"],
    "hyderabad": ["Hyderabadi Biryani", "Haleem", "Irani Chai", "Mirchi ka Salan"],
    "kolkata": ["Rosogolla", "Kathi Roll", "Hilsa Fish", "Mishti Doi"],
    "pune": ["Misal Pav", "Sabudana Khichdi", "Mastani"],
    "hassan": ["Akki Roti", "Ragi Mudde", "Kesari Bath"],
}
WEATHER_FOODS = {
    "rainy": ["hot pakoras", "masala chai", "Maggi noodles", "samosas"],
    "summer": ["watermelon juice", "coconut water", "mango lassi", "chaas"],
    "winter": ["hot chocolate", "gajar ka halwa", "peanut chikki", "warm khichdi"],
    "stormy": ["hot soup", "masala chai", "pakoras"],
    "clear skies": ["fresh lime soda", "cold coffee", "fruit chaat"],
    "partly cloudy": ["chai", "samosas", "bhel puri"],
}

def get_greeting(city, weather, season):
    import random
    city_lower = city.lower()
    temp = weather["temp"]
    desc = weather["desc"]
    icon = weather.get("icon", "😊")
    city_food = None
    for key, foods in CITY_FOODS.items():
        if key in city_lower:
            city_food = random.choice(foods)
            break
    weather_food = None
    if desc in WEATHER_FOODS:
        weather_food = random.choice(WEATHER_FOODS[desc])
    elif season in WEATHER_FOODS:
        weather_food = random.choice(WEATHER_FOODS[season])
    if desc == "rainy":
        opener = f"Hey! It's a lovely rainy day in {city} ({temp}°C) {icon}"
    elif desc == "clear skies" and temp > 30:
        opener = f"Hey! It's a hot sunny {temp}°C in {city} today {icon}"
    elif desc == "clear skies":
        opener = f"Hey! Beautiful clear skies in {city} today ({temp}°C) {icon}"
    elif desc == "stormy":
        opener = f"Hey! Stay safe — it's stormy in {city} right now {icon}"
    elif desc == "foggy":
        opener = f"Hey! Misty morning in {city} ({temp}°C) {icon}"
    else:
        opener = f"Hey! Hope you're enjoying the {desc} weather in {city} ({temp}°C) {icon}"
    if season == "summer" and temp > 32:
        tip = f"Stay hydrated! Perfect day for some {weather_food or 'coconut water'} 🥤"
    elif season == "winter":
        tip = f"Bundle up! Treat yourself to some {weather_food or 'warm chai'} ☕"
    elif desc == "rainy":
        tip = f"Nothing beats {weather_food or 'hot pakoras'} on a rainy day! 🍟"
    elif city_food:
        tip = f"Also — have you tried the {city_food} here? Absolutely must-try! 😋"
    else:
        tip = "Hope you're having a wonderful day!"
    return f"{opener}\n\n{tip}\n\nI'm your HR Policy Assistant — feel free to ask me anything about your HR policies! 📋"

GREETINGS = {"hi", "hello", "hey", "hii", "helo", "howdy", "sup", "yo", "greetings"}
SMALL_TALK = {
    "how are you", "how r u", "how are u", "what's up", "whats up",
    "good morning", "good afternoon", "good evening", "good night",
    "bye", "goodbye", "see you", "thanks", "thank you", "ok", "okay",
    "cool", "nice", "great", "awesome", "who are you", "what are you",
}

def is_small_talk(text):
    t = text.strip().lower().rstrip("!?.")
    return t in SMALL_TALK or t in GREETINGS or any(t.startswith(g) for g in GREETINGS)

def get_small_talk_reply(text, city, weather):
    t = text.strip().lower().rstrip("!?.")
    temp = weather["temp"]
    desc = weather["desc"]
    if t in GREETINGS or any(t.startswith(g) for g in GREETINGS):
        season = get_season(st.session_state.location.get("lat", 12.97))
        return get_greeting(city, weather, season)
    if "how are you" in t or "how r u" in t:
        return f"I'm doing great, thanks for asking! 😊 It's {desc} and {temp}°C in {city} — hope you're comfortable! How can I help you with HR policies today?"
    if "good morning" in t:
        return f"Good morning! 🌅 It's {temp}°C and {desc} in {city} right now. Hope you have a wonderful day ahead! Ask me anything about HR policies."
    if "good afternoon" in t:
        return f"Good afternoon! ☀️ A {desc} afternoon in {city} at {temp}°C. How can I assist you with HR policies?"
    if "good evening" in t:
        return f"Good evening! 🌙 It's {temp}°C in {city} tonight. How can I help you with HR policies?"
    if "good night" in t:
        return "Good night! 🌙 Rest well. Come back anytime you have HR policy questions!"
    if t in ("bye", "goodbye", "see you"):
        return f"Goodbye! 👋 Stay safe in the {desc} weather in {city}. Come back anytime!"
    if "thank" in t:
        return "You're welcome! 😊 Feel free to ask anytime."
    if "who are you" in t or "what are you" in t:
        return "I'm your HR Policy Assistant powered by IBM watsonx.ai! Upload your HR policy documents and I'll answer any questions from them. 📋"
    if t in ("ok", "okay", "cool", "nice", "great", "awesome"):
        return "😊 Let me know if you have any HR policy questions!"
    return "I'm here to help with your HR policy questions! Feel free to ask anything. 😊"

def clean_answer(raw):
    answer = raw
    stop_patterns = [
        r"\nQuestion[\s]*:", r"\nQ[\s]*:",
        r"\nAnswer[\s]*:", r"\nA[\s]*:",
        r"\n(Can|How|What|When|Where|Why|Is|Do|Will|Are|Should|May|Could)\s+\w",
        r"<\|user\|>", r"<\|system\|>", r"<\|assistant\|>",
    ]
    for pattern in stop_patterns:
        m = re.search(pattern, answer, re.IGNORECASE)
        if m:
            answer = answer[:m.start()]
    answer = answer.strip()
    if not answer:
        answer = "I could not find this in the HR policy documents."
    return answer

def load_file(uploaded_file):
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
            df = pd.read_excel(tmp_path, engine="openpyxl" if ext == "xlsx" else "xlrd")
            docs = [Document(page_content=" | ".join(f"{c}: {v}" for c, v in row.items() if str(v).strip() not in ("", "nan", "None")), metadata={"row": i, "source": uploaded_file.name}) for i, row in df.iterrows()]
            docs = [d for d in docs if d.page_content.strip()]
        elif ext == "csv":
            import pandas as pd
            df = pd.read_csv(tmp_path)
            docs = [Document(page_content=" | ".join(f"{c}: {v}" for c, v in row.items() if str(v).strip() not in ("", "nan", "None")), metadata={"row": i, "source": uploaded_file.name}) for i, row in df.iterrows()]
            docs = [d for d in docs if d.page_content.strip()]
        else:
            raise ValueError(f"Unsupported file type: .{ext}")
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        return docs
    finally:
        os.unlink(tmp_path)

# ── Location resolves from user input, defaults to Bengaluru ─────────
if "city_input" not in st.session_state:
    st.session_state.city_input = "Bengaluru"

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    # ── City input ────────────────────────────────────────────────────
    st.markdown("#### 📍 Your Location")
    city_input = st.text_input(
        "Enter your city",
        value=st.session_state.city_input,
        placeholder="e.g. Bengaluru",
        label_visibility="collapsed"
    )

    if city_input != st.session_state.city_input or st.session_state.location is None:
        st.session_state.city_input = city_input
        with st.spinner("Fetching weather..."):
            st.session_state.location = resolve_city(city_input)
            loc = st.session_state.location
            st.session_state.weather = get_weather(loc["lat"], loc["lon"])

    loc = st.session_state.location or {"city": "Bengaluru", "region": "Karnataka", "country": "India", "lat": 12.97, "lon": 77.59}
    w = st.session_state.weather or {"temp": 28, "desc": "pleasant", "icon": "🌤️", "humidity": 60, "wind": 10}

    # Weather card
    st.markdown(f"""
    <div class="weather-card">
        <div class="weather-city">📍 {loc['city']}, {loc['region']}</div>
        <div class="weather-temp">{w['icon']} {w['temp']}°C</div>
        <div class="weather-desc">{w['desc'].title()}</div>
        <div class="weather-row">
            <span>💧 {w['humidity']}%</span>
            <span>💨 {w['wind']} km/h</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    city = loc["city"]
    weather = w

    st.markdown("#### 📄 Upload Documents")
    st.caption("PDF · DOCX · TXT · XLSX · CSV")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            if st.button(f"📥 Index {len(new_files)} file(s)", use_container_width=True):
                all_chunks = []
                failed = []
                progress = st.progress(0, text="Starting...")
                for i, file in enumerate(new_files):
                    try:
                        progress.progress(int((i / len(new_files)) * 100), text=f"Loading {file.name}...")
                        docs = load_file(file)
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_documents(docs)
                        all_chunks.extend(chunks)
                        st.session_state.indexed_files.append(file.name)
                    except Exception as e:
                        failed.append(f"{file.name}: {str(e)}")
                if all_chunks:
                    progress.progress(90, text="Building vector index...")
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = FAISS.from_documents(all_chunks, embeddings)
                    else:
                        new_vs = FAISS.from_documents(all_chunks, embeddings)
                        st.session_state.vectorstore.merge_from(new_vs)
                    if st.session_state.model is None:
                        st.session_state.model = get_model()
                    progress.progress(100, text="Done!")
                    st.success(f"✅ Indexed {len(new_files)} file(s)")
                for f in failed:
                    st.error(f"❌ {f}")
        else:
            st.success("✅ All files indexed")

    if st.session_state.indexed_files:
        st.divider()
        st.markdown("**Indexed documents**")
        for fname in st.session_state.indexed_files:
            st.markdown(f'<span class="file-pill">📄 {fname}</span>', unsafe_allow_html=True)
        if st.button("🗑️ Clear all documents", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.indexed_files = []
            st.session_state.chat_history = []
            st.session_state.model = None
            st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("IBM watsonx.ai · granite-3-8b-instruct · eu-de")

# ── Chat UI ───────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#1e2d45;">
        <div style="font-size:48px;margin-bottom:16px;">📋</div>
        <div style="font-size:16px;font-weight:500;color:#374151;margin-bottom:8px;">No documents indexed yet</div>
        <div style="font-size:13px;color:#1e2d45;">Upload your HR policy documents from the sidebar to get started</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask about your HR policy or just say Hi! 👋"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if is_small_talk(question):
                answer = get_small_talk_reply(question, city, weather)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                with st.spinner("Asking Watson AI..."):
                    try:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                        docs = retriever.invoke(question)
                        context = "\n\n".join(doc.page_content for doc in docs)
                        sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))
                        excerpts = [doc.page_content[:100] for doc in docs]

                        system_msg = (
                            "You are an HR policy assistant. Answer ONE question in 1-3 sentences. "
                            "Use ONLY the context provided. Do NOT write more questions or answers after yours. "
                            "Stop after your answer. If not found in context say: "
                            "I could not find this in the HR policy documents."
                        )
                        prompt = (
                            f"<|system|>\n{system_msg}\n"
                            f"<|user|>\nContext:\n{context}\n\nQuestion: {question}\n"
                            f"<|assistant|>\n"
                        )

                        if st.session_state.model is None:
                            st.session_state.model = get_model()
                        raw = st.session_state.model.generate_text(prompt=prompt)
                        answer = clean_answer(raw)

                        st.markdown(answer)
                        with st.expander("📎 Sources used"):
                            st.caption(f"**Documents:** {', '.join(sources)}")
                            for i, exc in enumerate(excerpts, 1):
                                st.caption(f"{i}. {exc}…")

                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        err = f"❌ Watson AI error: {str(e)}"
                        st.error(err)
                        st.session_state.chat_history.append({"role": "assistant", "content": err})
