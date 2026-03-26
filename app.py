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

st.set_page_config(page_title="HR Policy Chatbot", page_icon="📋", layout="centered")
st.title("📋 HR Policy Chatbot")
st.caption("Powered by IBM watsonx.ai · Granite · RAG")

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

# ── Location + Weather (free, no API key) ─────────────────────────────
def get_location():
    """Get city from IP using ip-api.com (free, no key needed)."""
    try:
        r = requests.get("http://ip-api.com/json/", timeout=5)
        data = r.json()
        if data.get("status") == "success":
            return {
                "city": data.get("city", "your city"),
                "region": data.get("regionName", ""),
                "country": data.get("country", ""),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
            }
    except Exception:
        pass
    return {"city": "your city", "region": "", "country": "", "lat": 12.97, "lon": 77.59}

def get_weather(lat, lon):
    """Get current weather from Open-Meteo (free, no key needed)."""
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

        # WMO weather code to description
        if code == 0:
            desc = "clear skies"
        elif code in (1, 2):
            desc = "partly cloudy"
        elif code == 3:
            desc = "overcast"
        elif code in (45, 48):
            desc = "foggy"
        elif code in (51, 53, 55, 61, 63, 65):
            desc = "rainy"
        elif code in (71, 73, 75):
            desc = "snowy"
        elif code in (95, 96, 99):
            desc = "stormy"
        else:
            desc = "pleasant"

        return {
            "temp": round(temp),
            "desc": desc,
            "humidity": humidity,
            "code": code,
        }
    except Exception:
        pass
    return {"temp": 28, "desc": "pleasant", "humidity": 60, "code": 0}

def get_season(lat):
    """Determine season based on hemisphere and month."""
    month = datetime.now().month
    is_northern = lat >= 0
    if is_northern:
        if month in (12, 1, 2):
            return "winter"
        elif month in (3, 4, 5):
            return "spring"
        elif month in (6, 7, 8):
            return "summer"
        else:
            return "autumn"
    else:
        if month in (12, 1, 2):
            return "summer"
        elif month in (3, 4, 5):
            return "autumn"
        elif month in (6, 7, 8):
            return "winter"
        else:
            return "spring"

# Food suggestions by city (expand as needed)
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

def get_greeting(city: str, weather: dict, season: str) -> str:
    """Generate a warm, weather + food aware greeting."""
    city_lower = city.lower()
    temp = weather["temp"]
    desc = weather["desc"]

    # Pick city food
    city_food = None
    for key, foods in CITY_FOODS.items():
        if key in city_lower:
            import random
            city_food = random.choice(foods)
            break

    # Pick weather/season food
    import random
    weather_food = None
    if desc in WEATHER_FOODS:
        weather_food = random.choice(WEATHER_FOODS[desc])
    elif season in WEATHER_FOODS:
        weather_food = random.choice(WEATHER_FOODS[season])

    greetings = []

    # Weather-based openers
    if desc == "rainy":
        greetings.append(f"Hey! It's a lovely rainy day in {city} ({temp}°C) ☔")
    elif desc == "clear skies" and temp > 30:
        greetings.append(f"Hey! It's a sunny {temp}°C in {city} today ☀️")
    elif desc == "clear skies":
        greetings.append(f"Hey! Beautiful clear skies in {city} today ({temp}°C) 🌤️")
    elif desc == "stormy":
        greetings.append(f"Hey! Stay safe — it's stormy in {city} right now ⛈️")
    elif desc == "foggy":
        greetings.append(f"Hey! Misty morning in {city} ({temp}°C) 🌫️")
    else:
        greetings.append(f"Hey! Hope you're enjoying the {desc} weather in {city} ({temp}°C) 😊")

    # Season + food tip
    if season == "summer" and temp > 32:
        tip = f"Stay hydrated! Perfect day for some {weather_food or 'coconut water'} 🥤"
    elif season == "winter":
        tip = f"Bundle up! Treat yourself to some {weather_food or 'warm chai'} ☕"
    elif desc == "rainy":
        tip = f"Nothing beats {weather_food or 'hot pakoras'} on a rainy day! 🍟"
    elif city_food:
        tip = f"Also — have you tried the {city_food} here? Absolutely must-try! 😋"
    else:
        tip = f"Hope you're having a great day!"

    return f"{greetings[0]} {tip}\n\nI'm your HR Policy Assistant — feel free to ask me anything about your HR policies!"

# ── Detect non-HR / small talk ────────────────────────────────────────
GREETINGS = {"hi", "hello", "hey", "hii", "helo", "howdy", "sup", "yo", "greetings"}
SMALL_TALK = {
    "how are you", "how r u", "how are u", "what's up", "whats up",
    "good morning", "good afternoon", "good evening", "good night",
    "bye", "goodbye", "see you", "thanks", "thank you", "ok", "okay",
    "cool", "nice", "great", "awesome", "who are you", "what are you",
}

def is_greeting(text: str) -> bool:
    return text.strip().lower() in GREETINGS

def is_small_talk(text: str) -> bool:
    t = text.strip().lower().rstrip("!?.")
    return t in SMALL_TALK or any(t.startswith(g) for g in GREETINGS)

def get_small_talk_reply(text: str, city: str, weather: dict) -> str:
    t = text.strip().lower().rstrip("!?.")
    temp = weather["temp"]
    desc = weather["desc"]

    if any(t.startswith(g) for g in GREETINGS) or t in GREETINGS:
        season = get_season(st.session_state.location.get("lat", 12.97))
        return get_greeting(city, weather, season)

    if "how are you" in t or "how r u" in t or "how are u" in t:
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

# ── Answer cleaner ────────────────────────────────────────────────────
def clean_answer(raw: str) -> str:
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

# ── File loader ───────────────────────────────────────────────────────
def load_file(uploaded_file) -> list:
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
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        return docs
    finally:
        os.unlink(tmp_path)

# ── Fetch location + weather once per session ─────────────────────────
if st.session_state.location is None:
    with st.spinner("Detecting your location..."):
        st.session_state.location = get_location()

if st.session_state.weather is None and st.session_state.location:
    loc = st.session_state.location
    st.session_state.weather = get_weather(loc["lat"], loc["lon"])

city = st.session_state.location.get("city", "your city")
weather = st.session_state.weather or {"temp": 28, "desc": "pleasant", "humidity": 60, "code": 0}

# Show location in sidebar
st.sidebar.header("📄 Upload Documents")
st.sidebar.caption("PDF, DOCX, TXT, XLSX, XLS, CSV · Multiple files allowed")
if st.session_state.location:
    loc = st.session_state.location
    w = st.session_state.weather
    st.sidebar.info(
        f"📍 {loc['city']}, {loc['region']}\n\n"
        f"🌡️ {w['temp']}°C · {w['desc'].title()}\n\n"
        f"💧 Humidity: {w['humidity']}%"
    )

# ── File uploader ─────────────────────────────────────────────────────
uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt", "xlsx", "xls", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
    if new_files:
        if st.sidebar.button(f"📥 Index {len(new_files)} new file(s)"):
            all_chunks = []
            failed = []
            progress = st.sidebar.progress(0, text="Starting...")
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
                st.sidebar.success(f"✅ Indexed {len(new_files)} file(s)")
            if failed:
                for f in failed:
                    st.sidebar.error(f"❌ {f}")
    else:
        st.sidebar.success("✅ All uploaded files already indexed")

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

    if question := st.chat_input("Ask about your HR policy or just say Hi! 👋"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            # ── Small talk / greeting handler ──────────────────────────
            if is_small_talk(question):
                answer = get_small_talk_reply(question, city, weather)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # ── HR policy RAG handler ──────────────────────────────────
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
