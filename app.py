# app.py
import streamlit as st
from document_loader import fetch_google_doc, extract_doc_id
from vector_store import VectorStore
from chatbot import RAGChatbot
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Dark mode styling
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.user-message {
    background-color: #2b313e;
}
.bot-message {
    background-color: #1e2530;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'doc_loaded' not in st.session_state:
    st.session_state.doc_loaded = False

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("Enter Groq API Key", type="password", 
                            help="Get free key from console.groq.com")
    
    # Google Doc URL
    doc_url = st.text_input("Google Doc URL", 
                            placeholder="https://docs.google.com/document/d/...")
    
    if st.button("Load Document") and doc_url and api_key:
        with st.spinner("Loading document..."):
            try:
                doc_id = extract_doc_id(doc_url)
                content = fetch_google_doc(doc_id)
                
                # Initialize vector store
                st.session_state.vector_store = VectorStore()
                chunks_count = st.session_state.vector_store.ingest_document(
                    content, {"title": "User Document"}
                )
                
                # Initialize chatbot
                st.session_state.chatbot = RAGChatbot(
                    st.session_state.vector_store, 
                    api_key, 
                    use_groq=True
                )
                
                st.session_state.doc_loaded = True
                st.success(f"✅ Document loaded! Created {chunks_count} chunks.")
            except Exception as e:
                st.error(f"Error loading document: {str(e)}")
    
    if st.session_state.doc_loaded:
        st.info("🟢 Document ready for questions!")

# Main chat interface
st.title("🤖 RAG-Powered Document Chatbot")

if not st.session_state.doc_loaded:
    st.info("👈 Load a Google Doc from the sidebar to start chatting!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
