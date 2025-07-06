import streamlit as st
import tempfile
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

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

# Vector Store Setup
from langchain.vectorstores import FAISS

# Store the embedded documents in FAISS index
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Optional: Save to disk for reuse (you can skip this during prototyping)
# vectorstore.save_local("faiss_index")
st.write(f"✅ FAISS vector store created with {vectorstore.index.ntotal} documents.")
print("Chunks in index:", vectorstore.index.ntotal)
# Retriever Setup
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

query = "What topics are discussed in the document?"
docs = retriever.get_relevant_documents(query)

# LLM Setup - using Ollama
# Set up the LLM interface using Ollama
llm = OllamaLLM(model="llama2")  

prompt_template = """
You are a helpful assistant that answers questions based on the provided context.

Use ONLY the information from the context to answer. 
If you are unsure, say "I don't know" — do not make up an answer.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# RAG Chain Execution. This connects the following components from the above steps: retriever, llm, prompt
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# Sample user query
query = "What is the telework policy?"

# Run the full chain
response = qa_chain.invoke(query)
# Display
st.write("### ✅ Answer:")
st.write(response["result"])

# Optional: Show sources
for i, doc in enumerate(response["source_documents"]):
    st.write(f"**Source {i+1}:**")
    st.write(doc.page_content[:300])  # show first 300 chars

