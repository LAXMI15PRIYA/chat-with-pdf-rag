# 📄 Chat with PDF — Production-Style RAG Application

> An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and interact with their content using natural-language questions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)](https://faiss.ai/)
[![Groq](https://img.shields.io/badge/Groq-LLM_Inference-purple)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🔗 **Live Demo:** https://chat-with-pdf-rag-3bkjwjqw2sbphp5v8slzfn.streamlit.app/

🔗 **Repository:** https://github.com/LAXMI15PRIYA/chat-with-pdf-rag

---

## 📌 Overview

**Chat with PDF** is a Retrieval-Augmented Generation application designed to answer questions directly from user-provided PDF documents.

Instead of sending the entire document to a Large Language Model, the application first retrieves the most relevant sections of the document and provides only that context to the LLM.

This approach improves:

- Context relevance
- Response grounding
- Retrieval efficiency
- Scalability compared with processing an entire document for every question

The application implements the complete RAG workflow:

```text
PDF Upload
     ↓
Document Loading
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
FAISS Vector Index
     ↓
Semantic Retrieval
     ↓
Relevant Context
     ↓
Groq LLM
     ↓
Generated Answer
     ↓
Source Attribution
