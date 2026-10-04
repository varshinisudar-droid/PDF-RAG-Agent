# PDF Question Answering RAG + Tiny Agent

## Project Overview

This project is a **PDF Question Answering system using Retrieval-Augmented Generation (RAG)** combined with a **Tiny Agent**.

The system allows users to ask questions about information contained in PDF documents. The RAG system retrieves the most relevant information from the PDFs and uses Gemini to generate an answer based only on the retrieved content.

The Tiny Agent decides whether the user's request should be handled by:

* **Document Search** – for questions related to the PDF documents
* **Calculator** – for mathematical calculations

If the required information is not available in the PDFs, the system responds that the answer was not found in the provided documents.

---

## Project Architecture

```text
User Question
      |
      v
   Tiny Agent
      |
      +----------------------+
      |                      |
      v                      v
DOCUMENT SEARCH          CALCULATOR
      |                      |
      v                      v
   ChromaDB                Result
      |
      v
Retrieve Top-3 Chunks
      |
      v
     Gemini
      |
      v
Check Retrieved Information
      |
      +----------------------+
      |                      |
    FOUND                NOT FOUND
      |                      |
      v
```
