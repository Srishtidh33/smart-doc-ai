import numpy as np
import faiss
import ollama

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# -------------------------
# LOAD PDF
# -------------------------
reader = PdfReader("sample.pdf")

text = ""
for page in reader.pages:
    if page.extract_text():
        text += page.extract_text()

print("Document loaded")

# -------------------------
# CHUNKING
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# -------------------------
# EMBEDDINGS + FAISS
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index ready")

# -------------------------
# CHAT LOOP
# -------------------------
while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, 3)

    context = "\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
Answer using context only.

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model="mistral:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAI:", response['message']['content'])