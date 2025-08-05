from pathlib import Path
import faiss
import pickle

def verify_vectorstore():
    vs_path = Path("app/data/domains/hr/vectorstore")
    
    # Check files exist
    if not (vs_path/"index.faiss").exists():
        print("Missing index.faiss file")
        return False
    if not (vs_path/"index.pkl").exists():
        print("Missing index.pkl file")
        return False

    # Try loading the index
    try:
        index = faiss.read_index(str(vs_path/"index.faiss"))
        with open(vs_path/"index.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        print(f"Vectorstore contains {index.ntotal} vectors")
        print(f"Metadata has {len(metadata['chunks'])} documents")
        return True
    except Exception as e:
        print(f"Error loading vectorstore: {str(e)}")
        return False

if __name__ == "__main__":
    if verify_vectorstore():
        print("✅ Vectorstore is valid")
    else:
        print("❌ Vectorstore verification failed")