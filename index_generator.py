# ✅ index_generator.py

from document_loader import load_documents, split_documents
from vector_store import create_vector_store, save_vector_store

docs = load_documents("Docs")
chunks = split_documents(docs)
vs = create_vector_store(chunks)
save_vector_store(vs, "faiss_index")

