# Chat with PDF

A Streamlit app that lets you upload a PDF and ask questions about it — the AI answers using only the content of your uploaded document, no external paid API required.

## Features

- Upload PDF documents through a simple Streamlit interface
- Extract text from PDF files
- Split documents into smaller text chunks
- Generate semantic embeddings for each chunk
- Store and search embeddings using FAISS
- Retrieve the most relevant content based on your question
- Generate answers using a locally running FLAN-T5 model
- Runs completely locally and free — no paid LLM API needed

## Technologies Used

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PyPDF2
- FLAN-T5 Base

## How It Works

1. You upload a PDF through the Streamlit UI
2. The app extracts the text from the PDF
3. The text is split into smaller chunks
4. Each chunk is converted into a semantic embedding
5. Embeddings are stored in a FAISS vector index
6. When you ask a question, the app finds the most relevant chunks
7. Those chunks are passed to the FLAN-T5 model, which generates an answer based only on that content

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Sanskriti4200/chat-with-pdf.git
cd chat-with-pdf
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`), upload a PDF, and start asking questions.

## Project Structure

```
chat-with-pdf/
├── app.py          # Streamlit interface
├── model.py         # Model and embedding/retrieval logic
└── .gitignore
```

## Notes

- No API key or internet-based LLM service is required — everything runs locally.
- Answers are generated strictly from the content of the uploaded PDF.

## License

This project currently has no license specified.
