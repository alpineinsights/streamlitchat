import os
import requests
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
    """Client for Claude AI completions using direct API calls."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = CLAUDE_MODEL
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
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
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "system": system_prompt,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": f"""Here are some relevant documents to help answer the question:

{combined_contexts}

Based on the above documents, please answer this question: {query}"""
                }
            ]
        }
        
        try:
            # Store original proxy settings
            orig_http_proxy = os.environ.pop('HTTP_PROXY', None)
            orig_https_proxy = os.environ.pop('HTTPS_PROXY', None)
            orig_http_proxy_lower = os.environ.pop('http_proxy', None)
            orig_https_proxy_lower = os.environ.pop('https_proxy', None)
            
            # Make the API request with proxies explicitly disabled
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                proxies={"http": None, "https": None}
            )
            
            # Restore original proxy settings
            if orig_http_proxy:
