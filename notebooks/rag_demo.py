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

# Sidebar
st.sidebar.title("📂 Document Chat Assistant")
st.sidebar.markdown("Upload your PDF files here.")

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

# Load files
for file in uploads:
    temp_path = os.path.join(temp_dir.name, file.name)
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())

    loader = PyPDFLoader(temp_path)
    docs.extend(loader.load())

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(docs)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
st.write("🔄 Generating embeddings...")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# LLM via Ollama
llm = OllamaLLM(model="llama2")

# Prompt
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
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# ✅ Initialize chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 💬 UI: Question Input
st.header("📘 Ask a Question About Your Document")

query = st.text_input("Ask a question:", key="user_input", placeholder="e.g. What is the company travel policy?")

# 🧼 Optional: Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# 🧠 Handle new question
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(query)
        answer = response["result"]

        # Save to session history
        st.session_state.chat_history.append({"user": query, "bot": answer, "sources": response["source_documents"]})

# 💬 Display full conversation
for i, turn in enumerate(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {turn['user']}")
    st.markdown(f"**🤖 Assistant:** {turn['bot']}")

    # Expandable source for each answer
    with st.expander(f"📄 Sources used for Question {i+1}"):
        for j, doc in enumerate(turn["sources"]):
            st.markdown(f"**Source {j+1}**")
            st.write(doc.page_content[:500])
