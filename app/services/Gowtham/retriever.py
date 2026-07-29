import math
import re
from typing import List, Dict, Any, Optional

import config

class Document:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleInMemoryVectorStore:
    """
    A lightweight, robust in-memory vector store.
    Uses real embeddings (Gemini or OpenAI) if API keys are configured.
    Falls back to a keyword-matching scoring algorithm (TF-IDF/BM25 inspired)
    if API keys are missing, ensuring the pipeline remains runnable in any environment.
    """
    def __init__(self):
        self.documents: List[Document] = []
        self._embeddings_model = None
        self._initialized = False

    def initialize_embeddings(self, provider: str):
        """Initializes the embeddings client from LangChain."""
        api_key = config.get_api_key(provider)
        if not api_key:
            return False
            
        try:
            if provider == "gemini":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self._embeddings_model = GoogleGenerativeAIEmbeddings(
                    model=config.get_embed_model(provider),
                    google_api_key=api_key
                )
            elif provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                self._embeddings_model = OpenAIEmbeddings(
                    model=config.get_embed_model(provider),
                    api_key=api_key
                )
            self._initialized = True
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize {provider} embeddings: {e}. Falling back to keyword matching.")
            self._embeddings_model = None
            self._initialized = False
            return False

    def add_documents(self, documents: List[Document]):
        """Adds documents to the store."""
        self.documents.extend(documents)

    def _compute_keyword_score(self, doc_text: str, query: str) -> float:
        """Fallback TF-IDF style scoring helper for keyword search."""
        query_words = re.findall(r'\w+', query.lower())
        doc_words = re.findall(r'\w+', doc_text.lower())
        
        if not query_words or not doc_words:
            return 0.0
            
        score = 0.0
        doc_word_counts = {}
        for w in doc_words:
            doc_word_counts[w] = doc_word_counts.get(w, 0) + 1
            
        for qw in query_words:
            if qw in doc_word_counts:
                # Add score proportional to term frequency
                tf = doc_word_counts[qw] / len(doc_words)
                score += tf * (1.0 + math.log(1.0 + tf))
        return score

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Performs a search over the documents and returns the top k matches."""
        if not self.documents:
            return []
            
        provider = config.DEFAULT_PROVIDER
        # Try initializing embeddings dynamically if not yet done
        if not self._initialized:
            self.initialize_embeddings(provider)
            
        if self._initialized and self._embeddings_model:
            try:
                # Real semantic similarity search using cosine similarity over embeddings
                texts = [doc.page_content for doc in self.documents]
                query_vector = self._embeddings_model.embed_query(query)
                doc_vectors = self._embeddings_model.embed_documents(texts)
                
                # Compute cosine similarities
                scored_docs = []
                for i, doc_vec in enumerate(doc_vectors):
                    # Dot product / magnitude
                    dot_product = sum(a * b for a, b in zip(query_vector, doc_vec))
                    q_mag = math.sqrt(sum(a * a for a in query_vector))
                    d_mag = math.sqrt(sum(b * b for b in doc_vec))
                    similarity = dot_product / (q_mag * d_mag) if (q_mag * d_mag) > 0 else 0.0
                    scored_docs.append((similarity, self.documents[i]))
                    
                # Sort by score descending
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                return [doc for score, doc in scored_docs[:k]]
                
            except Exception as e:
                print(f"Error during semantic embedding search: {e}. Falling back to keyword search.")
                
        # Keyword-based search fallback
        scored_docs = []
        for doc in self.documents:
            score = self._compute_keyword_score(doc.page_content, query)
            scored_docs.append((score, doc))
            
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]
