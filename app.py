import streamlit as st
from vector_store import load_vector_store
from rag_claude_mixtral import RAGEngine

# Initialize session state variable
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Load vector store with caching
@st.cache_resource
def load_vector_store_cached():
    return load_vector_store("faiss_index")

# Load model
model_path = r"C:\Users\gauri\Desktop\Hyd Bot\Model\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
rag = RAGEngine(model_path, load_vector_store_cached())

# Page title
st.markdown(
    "<h2 style='display: flex; align-items: center;'>🧾 Toppobot - Product Chatbot</h2>",
    unsafe_allow_html=True
)

# --- Layout Styling for Input + Button ---
st.markdown("""
    <style>
        .input-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .stTextInput > div > div > input {
            height: 42px !important;
            font-size: 16px !important;
        }
        .stButton > button {
            height: 42px !important;
            font-size: 14px !important;
            padding: 0 16px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Input + Clear Button Row ---
with st.container():
    # Custom HTML/CSS alignment using columns
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Hi there! How can I assist you today?",
            value=st.session_state.user_input,
            key="user_input",
            label_visibility="visible",
            placeholder="Ask your questions"
        )

    def clear_input():
        st.session_state.user_input = ""
        st.experimental_rerun()

    with col2:
        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
        st.button("❌ Clear", on_click=clear_input)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Generate Response ---
if st.session_state.user_input.strip():
    with st.spinner("Generating response..."):
        answer = rag.query(st.session_state.user_input)
    st.success("Response:")
    st.write(answer)
