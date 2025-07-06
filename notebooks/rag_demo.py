import streamlit as st
import tempfile
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set page configuration
st.set_page_config(page_title="Chat with your Documents", layout="wide")

# Sidebar for navigation or future use (file upload will go here later)
st.sidebar.title("📂 Document Chat Assistant")
st.sidebar.markdown("Upload your PDF files here (coming soon).")

# App title
st.title("💬 Ask Questions About Your Documents")

# Text input box for user query
user_input = st.text_input("Ask a question:")

# Display input just for testing (temporary)
if user_input:
    st.write("You asked:", user_input)


# File uploader in the sidebar
uploads = st.sidebar.file_uploader(
    label="Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Stop execution until files are uploaded
if not uploads:
    st.sidebar.info("⬆️ Please upload one or more supported documents to continue.")
    st.stop()

# Temporary directory to store uploaded files
temp_dir = tempfile.TemporaryDirectory()
docs = []

# Loop through each uploaded file
for file in uploads:
    temp_path = os.path.join(temp_dir.name, file.name)
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())

    # Load and extract documents using PyPDFLoader
    loader = PyPDFLoader(temp_path)
    docs.extend(loader.load())

# Split text into chunks with overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_documents(docs)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed the chunks
st.write("🔄 Generating embeddings...")
embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
st.write(f"✅ Generated {len(embeddings)} vector embeddings.")

# Step 3.6: Vector Store Setup
from langchain.vectorstores import FAISS

# Store the embedded documents in FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)

# Optional: Save to disk for reuse (you can skip this during prototyping)
# vectorstore.save_local("faiss_index")
