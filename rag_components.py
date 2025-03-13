import requests
import anthropic
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Tuple, Optional

# Voyage AI model configurations
VOYAGE_EMBEDDING_MODEL = "voyage-large-2"
VOYAGE_RERANKER_MODEL = "voyage-reranker-v1"

# Claude model configuration
CLAUDE_MODEL = "claude-3-7-sonnet-20240229"

class VoyageAIClient:
    """Client for Voyage AI embedding and reranking services."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_endpoint = "https://api.voyageai.com/v1/embeddings"
        self.reranking_endpoint = "https://api.voyageai.com/v1/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def create_embeddings(self, text: str, model: str = VOYAGE_EMBEDDING_MODEL) -> Tuple[List[float], List[Tuple[int, float]]]:
        """
        Generate both dense and sparse embeddings using Voyage AI.
        
        Args:
            text: The input text to embed
            model: The embedding model to use
            
        Returns:
            Tuple of (dense_embedding, sparse_embedding)
        """
        payload = {
            "model": model,
            "input": text,
            "output_type": "hybrid"  # Request both dense and sparse embeddings
        }
        
        try:
            response = requests.post(
                self.embedding_endpoint,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract dense embedding vector
            dense_embedding = data['data'][0]['embedding']
            
            # Extract sparse embedding (indices and values)
            sparse_embedding = []
            if 'sparse_embedding' in data['data'][0]:
                sparse_data = data['data'][0]['sparse_embedding']
                indices = sparse_data['indices']
                values = sparse_data['values']
                sparse_embedding = list(zip(indices, values))
            
            return dense_embedding, sparse_embedding
            
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return [], []
    
    def rerank_documents(self, query: str, documents: List[str], model: str = VOYAGE_RERANKER_MODEL, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            model: The reranking model to use
            top_n: Number of top results to return
            
        Returns:
            List of reranked documents with scores
        """
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n
        }
        
        try:
            response = requests.post(
                self.reranking_endpoint,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return data['results']
            
        except Exception as e:
            st.error(f"Error reranking documents: {str(e)}")
            return []

class ClaudeClient:
    """Client for Claude AI completions."""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = CLAUDE_MODEL
    
    def generate_response(self, query: str, contexts: List[str], system_prompt: str = None) -> str:
        """
        Generate a response using Claude based on the query and retrieved contexts.
        
        Args:
            query: The user's question
            contexts: List of retrieved document texts
            system_prompt: Optional system prompt to customize Claude's behavior
            
        Returns:
            Claude's response
        """
        # Create combined context
        combined_contexts = "\n\n".join([f"DOCUMENT: {context}" for context in contexts])
        
        # If no system prompt is provided, use a default one
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer that question."
Keep your answers concise and to the point."""
        
        try:
            # Prepare the message
            message = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Here are some relevant documents to help answer the question:

{combined_contexts}

Based on the above documents, please answer this question: {query}"""
                    }
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            st.error(f"Error generating Claude response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."

class QdrantSearch:
    """Handles search operations with Qdrant."""
    
    def __init__(self, url: str, api_key: str, collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
    
    def hybrid_search(self, 
                      dense_vector: List[float], 
                      sparse_vector: List[Tuple[int, float]], 
                      limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both dense and sparse vectors.
        
        Args:
            dense_vector: Dense embedding vector
            sparse_vector: Sparse embedding as list of (index, value) tuples
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        # Convert sparse vector to the format Qdrant expects
        indices, values = zip(*sparse_vector) if sparse_vector else ([], [])
        
        # Create search request with hybrid search
        search_params = models.SearchParams(
            hnsw_ef=128,
            exact=False
        )
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=models.HybridVector(
                    dense=dense_vector,
                    sparse=models.SparseVector(
                        indices=list(indices),
                        values=list(values),
                    ) if sparse_vector else None
                ),
                search_params=search_params,
                limit=limit,
                with_payload=True
            )
            
            return results
            
        except Exception as e:
            st.error(f"Error searching Qdrant: {str(e)}")
            return []

class RAGChatbot:
    """RAG Chatbot combining all components."""
    
    def __init__(self, qdrant_url, qdrant_api_key, collection_name, claude_api_key, voyage_api_key):
        # Initialize Voyage AI client
        self.voyage_client = VoyageAIClient(voyage_api_key)
        
        # Initialize Claude client
        self.claude_client = ClaudeClient(claude_api_key)
        
        # Initialize Qdrant search
        self.qdrant_search = QdrantSearch(qdrant_url, qdrant_api_key, collection_name)
    
    def process_query(self, query: str, use_reranking: bool = True, top_k: int = 5) -> str:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: The user's question
            use_reranking: Whether to use reranking
            top_k: Number of documents to retrieve
            
        Returns:
            Generated response
        """
        # Step 1: Generate embeddings for the query
        dense_embedding, sparse_embedding = self.voyage_client.create_embeddings(query)
        
        if not dense_embedding:
            return "Failed to generate embeddings for your query. Please try again."
        
        # Step 2: Perform hybrid search in Qdrant
        search_results = self.qdrant_search.hybrid_search(
            dense_vector=dense_embedding,
            sparse_vector=sparse_embedding,
            limit=top_k
        )
        
        if not search_results:
            return "No relevant documents found to answer your question."
        
        # Extract document texts from search results
        documents = [result.payload.get("text", "") for result in search_results if "text" in result.payload]
        
        # Step 3: Optionally rerank documents
        if use_reranking and len(documents) > 1:
            reranked_results = self.voyage_client.rerank_documents(query, documents)
            if reranked_results:
                # Get the reranked document texts in the new order
                documents = [documents[result['index']] for result in reranked_results]
        
        # Step 4: Generate response with Claude
        response = self.claude_client.generate_response(query, documents)
        
        return response
