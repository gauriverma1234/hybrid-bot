# ✅ rag_claude_mixtral.py

from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
from langchain.prompts import PromptTemplate
from anthropic import Anthropic

class RAGEngine:
    def __init__(self, model_path, vector_store):
        self.llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
        self.db = vector_store

    def query(self, user_question, k=3):
        relevant_docs = self.db.similarity_search(user_question, k=k)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        # Prompt template to guide the model
        prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        ### System:
You are a friendly and knowledgeable product assistant for a product-based company. Your job is to help customers find the right product by using ONLY the information provided in the context. Be clear, polite, and helpful — like a smart shopping assistant.

### Context:
{context}

### Customer Question:
{question}

  
### Response Style:
Structure your answer in a way that's easy for customers to follow and understand:


---

**🔍 What You're Looking For:**  
Briefly restate what the customer is asking, in simple terms.

---

**📦 Products That Match:**  
List the products that match the customer's query.  
For each product, include:
- ✅ ** Product Name**
- 💡 **Description** (short summary of material and use)
- 💰 **Structure** (layers, coating types, etc.)
- ⭐ **Intended Application** (e.g., labels, wrappers, flexible packaging)

---

**🤔 Which One’s Best for You?**  
Offer a short, helpful comparison:
- Use bullet points or a table
- Highlight pros/cons or who it's best suited for
- Use friendly, non-technical language

---

**✅ My Recommendation:**  
Suggest the best match (or two), based on their needs.  
Phrase it like helpful advice, not a hard sell.

---

**📭 If I’m Missing Something:**  
If the info isn’t in the context, say:  
_"I'm sorry! I couldn’t find enough product info to fully answer that, but I’m here if you’d like to rephrase or ask something else!"_

---
"""
            )
        print(prompt_template.input_variables)
        prompt = prompt_template.format(context=context, question=user_question)  # Fill in your prompt variables
        response = self.llm(prompt, max_tokens=512, temperature=0.7)
        return response["choices"][0]["text"].strip()
