import streamlit as st

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
