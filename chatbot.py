# chatbot.py
import openai
from groq import Groq  # Alternative: faster & free tier

class RAGChatbot:
    def __init__(self, vector_store, api_key, use_groq=True):
        self.vector_store = vector_store
        self.conversation_history = []
        
        if use_groq:
            self.client = Groq(api_key=api_key)
            self.model = "llama-3.1-8b-instant"
        else:
            openai.api_key = api_key
            self.model = "gpt-4o-mini"
            self.use_groq = False
    
    def generate_response(self, user_query):
        """Generate answer with citations"""
        # Retrieve relevant chunks
        retrieval_results = self.vector_store.retrieve(user_query, top_k=3)
        
        if not retrieval_results['documents']:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Build context with citations
        context = ""
        for i, (doc, meta) in enumerate(zip(
            retrieval_results['documents'], 
            retrieval_results['metadatas']
        )):
            context += f"\n[Section {meta['chunk_id']}]: {doc}\n"
        
        # Construct prompt
        system_prompt = """You are a helpful assistant that answers questions based solely on the provided document context. 
        Always cite your sources using [Section X] format. If the information isn't in the context, say so clearly."""
        
        user_prompt = f"""Context from document:
{context}

User Question: {user_query}

Provide a concise answer with inline citations like [Section X]. If the answer isn't in the context, say "This information isn't in the document."
"""
        
        # Generate response
        if hasattr(self, 'use_groq') and not self.use_groq:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            answer = response.choices[0].message.content
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            answer = response.choices[0].message.content
        
        # Store conversation
        self.conversation_history.append({
            "query": user_query,
            "answer": answer,
            "context": retrieval_results
        })
        
        return answer
