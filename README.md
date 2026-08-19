# 📄 Chat with PDF — RAG Question Answering System

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF document and ask questions about its content using natural language.

The application extracts text from the PDF, splits it into chunks, generates semantic embeddings, stores them in a FAISS vector index, retrieves the most relevant chunks for a user's question, and uses a Groq-hosted LLM to generate a context-grounded answer.

🌐 **Live Demo:**  
https://chat-with-pdf-rag-3bkjwjqw2sbphp5v8slzfn.streamlit.app/

---

## 🚀 Project Overview

Traditional PDF search often depends on exact keyword matching.

This project uses **semantic search + Retrieval-Augmented Generation** to understand the meaning of a user's question and retrieve the most relevant information from the uploaded document.

The complete workflow is:

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Index
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Top Relevant Chunks
    ↓
Groq LLM
    ↓
Answer + Source Pages
