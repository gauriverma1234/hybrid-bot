# ✅ vector_store.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(docs):
    return FAISS.from_documents(docs, embeddings)

def save_vector_store(vectorstore, save_path):
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"✅ Saved FAISS vector store at {save_path}")

def load_vector_store(load_path):
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)