# vector_store.py
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class VectorStore:
    def __init__(self, openai_api_key=None):
        self.client = chromadb.Client()
        
        # Use OpenAI embeddings or Sentence Transformers
        if openai_api_key:
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
        else:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        self.collection = self.client.create_collection(
            name="doc_collection",
            embedding_function=self.embedding_fn
        )
    
    def chunk_text(self, text, chunk_size=800, chunk_overlap=200):
        """Split text into semantic chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    
    def ingest_document(self, text, doc_metadata):
        """Add document chunks to vector store"""
        chunks = self.chunk_text(text)
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_metadata.get("title", "Unknown"), 
                      "chunk_id": i} for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)
    
    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant chunks"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
