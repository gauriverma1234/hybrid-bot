# ✅ document_loader.py

import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def clean_text(text):
    lines = text.split("\n")
    cleaned = [line.strip() for line in lines if line.strip() and is_english(line)]
    return " ".join(cleaned)

def load_documents(doc_dir):
    documents = []
    for filename in os.listdir(doc_dir):
        full_path = os.path.join(doc_dir, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(full_path)
            elif filename.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(full_path)
            else:
                continue

            for doc in loader.load():
                cleaned = clean_text(doc.page_content)
                if cleaned.strip():
                    doc.page_content = cleaned
                    documents.append(doc)
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

    print(f"✅ Loaded {len(documents)} documents.")
    return documents

def split_documents(documents, chunk_size=200, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, # Tune this size
        chunk_overlap=80, # Ensures continuity
        separators=["\n\n", "\n", ".", " "]  # Intelligent breaking points
    )
    chunks = splitter.split_documents(documents)
    return chunks
