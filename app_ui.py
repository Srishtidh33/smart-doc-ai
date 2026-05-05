import streamlit as st
import numpy as np
import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import ollama

from rag_engine import RAGEngine

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Smart RAG AI", layout="wide")

st.title("📄 Smart RAG AI (ChatGPT Style)")
st.write("Chat with your PDF or ask general questions.")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

rag = RAGEngine(model)

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ready" not in st.session_state:
    st.session_state.ready = False

# -------------------------
# PDF UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload PDF (optional)", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    rag.build_index(chunks)

    st.session_state.ready = True
    st.success("PDF loaded and indexed!")

# -------------------------
# CHAT HISTORY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
query = st.chat_input("Ask something...")

if query:

    # Save user msg
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # -------------------------
    # RETRIEVAL (RAG)
    # -------------------------
    if st.session_state.ready:
        top_chunks = rag.search(query, top_k=3)
        context = "\n".join(top_chunks)
    else:
        context = "No document uploaded. Answer generally."

    # -------------------------
    # PROMPT
    # -------------------------
    prompt = f"""
You are a helpful AI assistant.

Use context if available.

Context:
{context}

Question:
{query}

Answer clearly:
"""

    # -------------------------
    # AI RESPONSE
    # -------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            response = ollama.chat(
                model="mistral:latest",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"]

            st.markdown(answer)

    # Save assistant msg
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )