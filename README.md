# 🤖 RAG Document Chatbot

AI-powered chatbot that reads Google Docs and answers questions with citations using Retrieval-Augmented Generation.

## 🚀 Live Demo
- **Deployed App:** [https://doc-reader-chatbot-xnduqvcdtsqisyyzi5rvya.streamlit.app/](https://doc-reader-chatbot-xnduqvcdtsqisyyzi5rvya.streamlit.app/)
- **Demo Video:** [https://drive.google.com/file/d/1TnFcu1ogMgEr8fDjHCJue9gCqyBxcbSP/view?usp=sharing](https://drive.google.com/file/d/1TnFcu1ogMgEr8fDjHCJue9gCqyBxcbSP/view?usp=sharing)
- **GitHub Repo:** https://github.com/SwayamSiddhiDas/doc-reader-chatbot

## ✨ Features
- ✅ Fetches content from public Google Docs
- ✅ Semantic search with ChromaDB vector database
- ✅ LLM-powered answers via Groq (Llama 3.1)
- ✅ Inline citations [Section X] in responses
- ✅ Handles edge cases and irrelevant queries
- ✅ Multi-turn conversation support

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM:** Groq (Llama-3.1-8b-instant)
- **Vector DB:** ChromaDB
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Hosting:** Streamlit Cloud (Free)

## 📦 Installation

```bash
git clone https://github.com/SwayamSiddhiDas/doc-reader-chatbot.git
cd doc-reader-chatbot
pip install -r requirements.txt
python -m streamlit run app.py
```

## 🎯 How to Use
1. Get free Groq API key from console.groq.com
2. Create a public Google Doc (Share → Anyone with link)
3. Open the app and paste API key + Doc URL in sidebar
4. Click "Load Document" and start asking questions!

## 🧪 Test Queries
- "What is this document about?"
- "What are the key points in section 2?"
- "What is the price of [product]?"
- "Do you sell [item not in doc]?" (edge case test)

## 📊 Performance
- **Accuracy:** 90%+ on test queries
- **Response Time:** 2-5 seconds
- **Chunk Size:** 600 tokens with 100 overlap
- **Retrieval:** Top-5 relevant chunks per query

## 📂 Project Structure
```
doc-reader-chatbot/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## 🚀 Deployment
Deployed on Streamlit Cloud with free tier. Auto-deploys on git push.
