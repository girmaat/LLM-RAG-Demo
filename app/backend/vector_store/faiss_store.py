import os
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

class FAISSRetriever(BaseRetriever):
    """Complete debugged implementation of FAISS retriever"""
    
    # Pydantic configuration to allow arbitrary attributes
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(
        self, 
        index: Any,  # faiss.Index type causes Pydantic issues
        embedder: Any,
        metadata: Dict[str, Any],
        search_kwargs: Optional[Dict] = None
    ):
        super().__init__()
        print(f"🐞 [FAISSRetriever] Initializing with index: {type(index)}")  # Debug
        
        # Manually set attributes (bypass Pydantic validation)
        object.__setattr__(self, 'index', index)
        object.__setattr__(self, 'embedder', embedder)
        object.__setattr__(self, 'metadata', metadata)
        object.__setattr__(self, 'search_kwargs', search_kwargs or {'k': 3})
        
        print("🐞 [FAISSRetriever] Initialization complete")  # Debug

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Debugged retrieval method"""
        try:
            print(f"🐞 [Retrieval] Processing query: {query[:50]}...")  # Debug
            
            # 1. Generate embedding
            print("🐞 [Retrieval] Generating embedding...")
            embedding = self.embedder.embed_query(query)
            
            # 2. Perform search
            k = self.search_kwargs.get('k', 3)
            print(f"🐞 [Retrieval] Searching with k={k}...")
            distances, indices = self.index.search(
                np.array([embedding], dtype='float32'), 
                k
            )
            print(f"🐞 [Retrieval] Found {len(indices[0])} results")  # Debug
            
            # 3. Apply score threshold
            score_threshold = self.search_kwargs.get('score_threshold')
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if score_threshold is None or dist <= score_threshold:
                    doc = self.metadata['chunks'][idx]
                    doc.metadata['score'] = float(dist)
                    results.append(doc)
                    print(f"🐞 [Result {i+1}] Score: {dist:.3f}, Page: {doc.metadata.get('page_number', '?')}")
            
            return results
            
        except Exception as e:
            print(f"❌ [Retrieval Error] {str(e)}")
            return []

def load_faiss_index(
    embedder: Any,
    persist_path: str,
    search_kwargs: Optional[Dict] = None
) -> FAISSRetriever:
    """Debugged FAISS index loader"""
    print(f"🐞 [FAISS] Loading from: {persist_path}") 
    print(f"🐞 [FAISS] Expected files: {Path(persist_path)/'index.faiss'} and {Path(persist_path)/'index.pkl'}")
    
    index_path = Path(persist_path) / "index.faiss"
    meta_path = Path(persist_path) / "index.pkl"
    
    # 1. Validate paths
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    print("🐞 [load_faiss_index] Paths validated")

    # 2. Load index and metadata
    try:
        print("🐞 [load_faiss_index] Loading FAISS index...")
        index = faiss.read_index(str(index_path))
        
        print("🐞 [load_faiss_index] Loading metadata...")
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        
        print(f"🐞 [load_faiss_index] Loaded {len(metadata['chunks'])} chunks")  # Debug
        
        # 3. Create retriever
        print("🐞 [load_faiss_index] Creating retriever...")
        return FAISSRetriever(
            index=index,
            embedder=embedder,
            metadata=metadata,
            search_kwargs=search_kwargs or {'k': 3, 'score_threshold': 0.85}
        )
    except Exception as e:
        print(f"❌ [load_faiss_index] Failed: {str(e)}")
        raise


def build_faiss_index(
    documents: List[Document],
    embedder: Any,
    domain_name: str,
    index_name: str = "index"
) -> None:
    print(f"\n=== DEBUG: Path Verification ===")
    persist_path = Path("app") / "data" / "domains" / domain_name / "vectorstore"
    print(f"Relative path: {persist_path}")
    print(f"Absolute path: {persist_path.absolute()}")
    print(f"Parent exists: {persist_path.parent.exists()}")
    
    # Create directory if needed
    persist_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory created: {persist_path.exists()}")
    """Build and persist FAISS index from documents"""
    print(f"🐞 [build_faiss_index] Building index for {len(documents)} documents")
    
    try:
        # 1. Create embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = embedder.embed_documents(texts)
        
        # 2. Create and build index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype='float32'))
        
        # 3. Prepare metadata
        metadata = {
            'chunks': documents,
            'embedder': embedder.model_name,
            'dimension': dimension
        }
        
        # 4. Create directory structure (fixed path)
        persist_path = Path("app") / "data" / "domains" / domain_name / "vectorstore"
        print(f"🐞 [FAISS] Writing to: {persist_path}")
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # After writing the files, add:
        print("\n=== File Write Verification ===")
        print(f"index.faiss exists: {(persist_path / 'index.faiss').exists()}")
        print(f"index.pkl exists: {(persist_path / 'index.pkl').exists()}")


        # 5. Persist to disk
        faiss.write_index(index, str(persist_path / f"{index_name}.faiss"))
        with open(persist_path / f"{index_name}.pkl", "wb") as f:
            pickle.dump(metadata, f)
        print(f"🐞 [FAISS] Files created: {os.listdir(persist_path)}")   
        print(f"✅ Saved FAISS index to {persist_path}")
        
    except Exception as e:
        print(f"❌ Failed to build index: {str(e)}")
        raise