from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # New recommended import
except ImportError:
    # Fallback for older versions
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import warnings
    warnings.warn(
        "Using deprecated HuggingFaceEmbeddings. Install langchain-huggingface for the updated version.",
        DeprecationWarning
    )

def split_into_chunks(docs, chunk_size=500, chunk_overlap=100):
    """Split documents while preserving metadata"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
        add_start_index=True  # Helps track original positions
    )
    
    chunks = []
    for doc in docs:
        try:
            # Preserve all original metadata
            new_chunks = splitter.split_documents([doc])
            for chunk in new_chunks:
                chunk.metadata = doc.metadata.copy()  # Copy all metadata
                # Add chunk-specific info
                chunk.metadata.update({
                    "chunk_id": len(chunks),
                    "is_chunk": True,
                    "chunk_size": len(chunk.page_content)  # Add character count
                })
            chunks.extend(new_chunks)
        except Exception as e:
            print(f"Error splitting document: {e}")
            continue
    
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    return chunks

def get_embedder():
    """Returns embedding model with updated import"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32  # Added for better performance
        }
    )

# Test function for debugging
def _test_embeddings():
    """Verify the embedder produces numpy arrays"""
    embedder = get_embedder()
    test_texts = ["This is a test", "Another test"]
    embeddings = embedder.embed_documents(test_texts)
    print(f"Embedding type: {type(embeddings)}")
    if isinstance(embeddings, list):
        print(f"Converted to numpy array: {np.array(embeddings).shape}")

if __name__ == "__main__":
    import numpy as np
    _test_embeddings()