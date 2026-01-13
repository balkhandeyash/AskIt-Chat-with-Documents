import streamlit as st
import os
import requests
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# -----------------------------
# ANSWER QUESTIONS
# -----------------------------
def answer_question(vectorstore, query):
    try:
        # 1. Retrieve Context
        count = vectorstore._collection.count()
        docs = vectorstore.similarity_search(query, k=min(3, count) if count > 0 else 1)
        context = "\n".join([doc.page_content for doc in docs])

        # 2. Updated stable endpoint for the router
        # Note: the '/hf-inference' part is often redundant in the newest v1 path
        API_URL = "https://router.huggingface.co/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # 3. Payload with a guaranteed "warm" model
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct", 
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Use ONLY the provided context to answer."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Check if the response is actually JSON before parsing
        if response.status_code == 200:
            try:
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.JSONDecodeError:
                return f"Decoding Error: Received non-JSON response: {response.text[:200]}"
        else:
            return f"Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"System Error: {str(e)}"
# -----------------------------
# CACHE THE EMBEDDINGS
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

# -----------------------------
# PROCESS INPUT DOCUMENTS
# -----------------------------
def process_input(input_type, input_data):
    documents = ""
    splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=70)

    if input_type == "Link":
        all_docs = []
        for link in input_data:
            loader = WebBaseLoader(link)
            all_docs.extend(loader.load())
        texts = [doc.page_content for doc in splitter.split_documents(all_docs)]
    else:
        if input_type == "PDF":
            for file in input_data:
                reader = PdfReader(BytesIO(file.read()))
                for page in reader.pages:
                    documents += page.extract_text() or ""
        elif input_type == "DOCX":
            for file in input_data:
                doc = Document(BytesIO(file.read()))
                documents += "\n".join(p.text for p in doc.paragraphs)
        elif input_type == "TXT":
            for file in input_data:
                documents += file.read().decode("utf-8")
        elif input_type == "Text":
            documents = input_data

        texts = splitter.split_text(documents)

    embeddings = load_embeddings()
    return Chroma.from_texts(texts, embeddings)

# -----------------------------
# STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config(page_title="AskIt", layout="wide")
    st.title("AskIt – Chat with Documents")

    with st.sidebar:
        st.header("Setup")
        if HUGGINGFACE_API_TOKEN:
            st.success("API Token Connected")
        else:
            st.error("Missing HUGGINGFACEHUB_API_TOKEN in .env")

    input_type = st.selectbox("Select Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    input_data = None
    if input_type == "Link":
        n = st.number_input("Number of links", 1, 5, 1)
        input_data = [st.text_input(f"Link {i+1}", key=f"lnk_{i}") for i in range(n)]
        input_data = [l for l in input_data if l.strip()]
    elif input_type == "Text":
        input_data = st.text_area("Paste your text here")
    else:
        input_data = st.file_uploader(f"Upload {input_type} files", type=[input_type.lower()], accept_multiple_files=True)

    if st.button("Process Documents"):
        if not input_data:
            st.warning("Please provide input first.")
        else:
            with st.spinner("Processing..."):
                st.session_state.vectorstore = process_input(input_type, input_data)
                st.success("Ready to chat!")

    if "vectorstore" in st.session_state:
        st.divider()
        query = st.text_input("Ask a question about your content:")
        if st.button("Ask AI"):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing context..."):
                    answer = answer_question(st.session_state.vectorstore, query)
                    st.info("**Answer:**")
                    st.write(answer)

if __name__ == "__main__":
    main()