# Smart Doc AI

Chat with your PDF documents like ChatGPT using RAG (Retrieval-Augmented Generation), embeddings, and local LLMs.

# Overview

Smart Doc AI is a document-based AI assistant that allows you to upload PDFs and ask natural language questions about them. It uses a RAG pipeline to retrieve relevant context from documents and generate accurate answers using a local LLM (Ollama).

Instead of manually searching through PDFs, you can simply chat with your document.

# Features
📄 Upload and process PDF documents
🔍 Intelligent text chunking for better retrieval
🧠 Semantic search using embeddings
⚡ RAG-based question answering system
💬 ChatGPT-style Streamlit UI
🏠 Works fully locally using Ollama (no API cost)
📚 Context-aware responses from documents
🧠 How It Works (Architecture)
PDF Upload
   ↓
Text Extraction
   ↓
Chunking (LangChain)
   ↓
Embedding Generation (Sentence Transformers)
   ↓
Similarity Search (Cosine / FAISS)
   ↓
Top-K Relevant Chunks
   ↓
Prompt + Context → Ollama LLM
   ↓
Final Answer

# Tech Stack
Python 🐍
Streamlit 🎈
Sentence Transformers 🤖
LangChain (Text Splitters) 🔗
Ollama (Mistral / Phi models) 🧠
Scikit-learn 📊
PyPDF 📄

📂 Project Structure
smart-doc-ai/
│
├── app_ui.py          # Streamlit ChatGPT-style UI
├── app.py             # CLI version (RAG pipeline)
├── sample.pdf        # Test document
├── requirements.txt  # Dependencies
└── README.md

# Installation
1. Clone Repository
git clone https://github.com/yourusername/smart-doc-ai.git
cd smart-doc-ai
2. Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows
3. Install Dependencies
pip install -r requirements.txt
4. Install & Run Ollama

# Install Ollama:
👉 https://ollama.com

Then run a model:

ollama run mistral
5. Run Streamlit App
streamlit run app_ui.py

# Example Use Cases
Summarize long PDFs
Ask questions from notes/books
Understand research papers
Extract key information quickly
📸 UI Preview

<img width="1792" height="750" alt="image" src="https://github.com/user-attachments/assets/8ac1d68b-1e94-4856-8575-f89e58c16c68" />
assets/ui.png

📈 Future Improvements
🔥 Add FAISS / ChromaDB vector database
🧠 Multi-document chat support
💾 Chat memory (conversation history)
🌐 Deploy on cloud (Streamlit Cloud / Render)
📱 Mobile-friendly UI
👩‍💻 Author

Srishti Dhillon
AI & Software Developer Enthusiast
Passionate about AI, GenAI, RAG systems, and building real-world AI apps.

⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!
