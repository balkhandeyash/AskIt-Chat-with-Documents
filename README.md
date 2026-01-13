
# AskIt – AI Chat with Documents 🚀

**AskIt** is a high-performance Retrieval-Augmented Generation (RAG) application that allows users to interact with their documents and web content through natural language. Using state-of-the-art embeddings and Large Language Models (LLMs), AskIt provides context-aware answers based strictly on your provided data.



---

## ✨ Key Features

- **Multi-Source Ingestion**: Seamlessly upload **PDF, DOCX, TXT**, and raw text snippets.
- **Web Intelligence**: Extract and query content directly from any **web URL**.
- **Vector Intelligence**: Uses **ChromaDB** for efficient semantic search and document retrieval.
- **Persistent Chat History**: Integrated session tracking with a "Hide/Unhide" dashboard in the sidebar.
- **Optimized LLM**: Powered by **Qwen 2.5 (7B Instruct)** via Hugging Face Inference API for fast and accurate responses.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **RAG Framework**: [LangChain](https://www.langchain.com/)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Models**: [Hugging Face Inference API](https://huggingface.co/inference-api)

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and set up a clean environment:

```bash
git clone [https://github.com/balkhandeyash/Chat-With-Documents.git](https://github.com/balkhandeyash/Chat-With-Documents.git)
cd Chat-With-Documents
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Configuration

Create a `.env` file in the root directory and add your API credentials:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

```

### 3. Usage

Launch the local development server:

```bash
streamlit run app.py

```

---

## 📖 How It Works

1. **Chunking**: Input text is split into segments of 700 characters with overlap to preserve context.
2. **Indexing**: Text chunks are converted into 768-dimensional vectors and stored in a local **Chroma** collection.
3. **Retrieval**: When a user asks a question, the system performs a semantic search to find the most relevant context.
4. **Synthesis**: The AI (Qwen 2.5) analyzes the retrieved context to generate a precise answer without "hallucinating" external info.

---

## 📂 Project Structure

```text
.
├── app.py              # Main Streamlit application logic
├── requirements.txt    # Python dependencies
├── .env                # API Keys (Excluded from Git)
├── .gitignore          # Rules for Git (ignores venv/ and .env)
└── README.md           # Project documentation

```

---

## 🛡️ License & Credits

* **Author**: Yash Balkhande
* **Acknowledgment**: Built using the [LangChain](https://github.com/langchain-ai/langchain) framework and [Hugging Face](https://huggingface.co/) models.

---
