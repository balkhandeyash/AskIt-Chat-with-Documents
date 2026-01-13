# Chat-With-Documents

This is Q&A application that allows users to upload and interact with multiple types of documents (PDF, DOCX, TXT, etc.) or provide links to webpages. The application uses Retrieval-Augmented Generation (RAG) to process the documents, index them, and answer questions based on their content.

This project leverages Langchain and Hugging Face to create a powerful and flexible solution for querying documents. It uses FAISS to create an index of document embeddings and then allows users to query those documents through a chatbot-like interface.

---

## Features

- **Multi-file support**: Upload multiple documents at once in PDF, DOCX, TXT, or CSV format.
- **Web links**: Provide links to webpages, and AskIt will extract the text and allow you to query it.
- **Text-based query system**: Ask questions related to the documents or links, and get accurate answers based on the content.
- **Question-Answer history**: View a history of all previous questions and answers in the session.
- **Powered by Hugging Face models**: The application uses `Mistral-7B-Instruct` for answering questions.

---

## Requirements

To run this application, you will need the following:

- Python 3.7+
- Streamlit
- Hugging Face API Key
- Dependencies specified in `requirements.txt`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SqwareInfotechLearn/Chat-With-Documents.git
cd Chat-With-Documents
```

### 2. Set up the environment

Make sure you have Python 3.7 or higher installed. You can use `virtualenv` or `conda` to create an isolated environment.

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/MacOS
```

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Hugging Face API Key

To use the Hugging Face model, you'll need an API key. You can get it by creating an account on [Hugging Face](https://huggingface.co/).

After obtaining the API key, create a `.env` file in the root directory and add your key:

```
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

---

## Running the App

After setting up the environment and API key, you can run the app with the following command:

```bash
streamlit run app.py
```

This will start the Streamlit app and open it in your browser.

---

## How It Works

1. **Upload Documents**: Users can upload multiple files in various formats (PDF, DOCX, TXT, etc.) or provide links to webpages.
2. **Document Processing**: The application processes the documents using `langchain` and converts them into embeddings using a Hugging Face model.
3. **Searchable Index**: The documents are indexed using FAISS, allowing fast retrieval of relevant information.
4. **Ask Questions**: Once the documents are processed, users can ask questions, and the application will provide answers based on the content of the documents or links.
5. **Question History**: All questions and answers are saved in a session history and can be viewed at any time.

---

## File Formats Supported

- **PDF**: Upload PDF documents and ask questions about their content.
- **DOCX**: Upload DOCX documents and interact with the text.
- **TXT**: Upload plain text files for querying.
- **CSV**: Upload CSV files, and the application will treat the data as text for querying.
- **Web Links**: Provide links to webpages, and the application will extract text from the page for querying.

---

## Acknowledgments

- [Langchain](https://www.langchain.com/) for the document processing pipeline.
- [Hugging Face](https://huggingface.co/) for providing the transformer models.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
- [Streamlit](https://streamlit.io/) for providing an easy way to build interactive apps.
# AskIt-Chat-with-Documents
