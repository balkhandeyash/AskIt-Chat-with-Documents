import streamlit as st
import os
import requests
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
os.environ["USER_AGENT"] = "MyChatApp/1.0"
load_dotenv()
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # To store list of {question, answer}

# -----------------------------
# ANSWER QUESTIONS
# -----------------------------
def answer_question(vectorstore, query):
    try:
        count = vectorstore._collection.count()
        docs = vectorstore.similarity_search(query, k=min(3, count) if count > 0 else 1)
        context = "\n".join([doc.page_content for doc in docs])

        API_URL = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct", 
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use ONLY the provided context to answer."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            "max_tokens": 512,
            "temperature": 0.1
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
            # SAVE TO HISTORY
            st.session_state.chat_history.append({"question": query, "answer": answer})
            return answer
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"System Error: {str(e)}"

# -----------------------------
# CACHE THE EMBEDDINGS & PROCESSOR
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
        
        # --- NEW HISTORY SECTION ---
        st.divider()
        st.subheader("📜 Chat History")
        if not st.session_state.chat_history:
            st.info("No questions asked yet.")
        else:
            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.rerun()
            
            # Display history in expanders (Hide/Unhide functionality)
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question'][:30]}..."):
                    st.write(f"**Question:** {chat['question']}")
                    st.write(f"**Answer:** {chat['answer']}")

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