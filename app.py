import streamlit as st
import requests
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

# ============ DOCUMENT LOADER ============
def extract_doc_id(url):
    if '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    return url

def fetch_google_doc(doc_id):
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url)
    if response.status_code == 200:
        return response.text
    raise Exception(f"Failed to fetch document. Status: {response.status_code}")

# ============ VECTOR STORE ============
class VectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.create_collection(
            name="doc_collection",
            embedding_function=self.embedding_fn
        )
    
    def chunk_text(self, text, chunk_size=800, chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    
    def ingest_document(self, text, doc_metadata):
        chunks = self.chunk_text(text)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_metadata.get("title", "Unknown"), 
                      "chunk_id": i} for i in range(len(chunks))]
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)
    
    def retrieve(self, query, top_k=3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }

# ============ CHATBOT ============
class RAGChatbot:
    def __init__(self, vector_store, api_key):
        self.vector_store = vector_store
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        self.conversation_history = []
    
    def generate_response(self, user_query):
        try:
            retrieval_results = self.vector_store.retrieve(user_query, top_k=3)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
        
        if not retrieval_results['documents']:
            return "I couldn't find relevant information in the document."
        
        context = ""
        for doc, meta in zip(retrieval_results['documents'], retrieval_results['metadatas']):
            context += f"\n[Section {meta['chunk_id']}]: {doc}\n"
        
        system_prompt = """Answer questions based solely on the provided context. 
Always cite sources using [Section X]. If info isn't in context, say so clearly."""
        
        user_prompt = f"""Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer with inline citations [Section X]:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ============ STREAMLIT UI ============
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.markdown("""<style>.stApp {background-color: #0e1117;}</style>""", unsafe_allow_html=True)

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'doc_loaded' not in st.session_state:
    st.session_state.doc_loaded = False

with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", help="Get from console.groq.com")
    doc_url = st.text_input("Google Doc URL", placeholder="https://docs.google.com/document/d/...")
    
    if st.button("Load Document") and doc_url and api_key:
        with st.spinner("Loading..."):
            try:
                doc_id = extract_doc_id(doc_url)
                content = fetch_google_doc(doc_id)
                st.session_state.vector_store = VectorStore()
                chunks_count = st.session_state.vector_store.ingest_document(
                    content, {"title": "User Document"}
                )
                st.session_state.chatbot = RAGChatbot(st.session_state.vector_store, api_key)
                st.session_state.doc_loaded = True
                st.success(f"✅ Loaded! {chunks_count} chunks created.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.doc_loaded:
        st.info("🟢 Ready!")

st.title("🤖 RAG Document Chatbot")

if not st.session_state.doc_loaded:
    st.info("👈 Load a Google Doc to start!")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})


