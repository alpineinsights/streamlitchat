import os
import requests
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Tuple, Optional

# Voyage AI model configurations
VOYAGE_EMBEDDING_MODEL = "voyage-large-2"  # Default model, can be overridden
VOYAGE_RERANKER_MODEL = "rerank-2"  # Latest reranker model

# Claude model configuration
CLAUDE_MODEL = "claude-3-7-sonnet-20240229"

class VoyageAIClient:
    """Client for Voyage AI embedding and reranking services."""
    
    def __init__(self, api_key: str):
        """Initialize the Voyage AI client with API key."""
        self.api_key = api_key
        self.embedding_endpoint = "https://api.voyageai.com/v1/embeddings"
        self.reranking_endpoint = "https://api.voyageai.com/v1/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def create_embeddings(self, text: str, model: str = VOYAGE_EMBEDDING_MODEL, output_type: str = "dense") -> Tuple[List[float], List[Tuple[int, float]]]:
        """
        Generate embeddings using Voyage AI.
        
        Args:
            text: The input text to embed
            model: The embedding model to use (voyage-large-2, voyage-finance-2, etc.)
            output_type: Type of embedding to generate ('hybrid', 'dense', or 'sparse')
            
        Returns:
            Tuple of (dense_embedding, sparse_embedding)
        """
        # Prepare the request payload
        payload = {
            "model": model,
            "input": text
        }
        
        # Only add output_type for hybrid requests with models that support it
        if output_type == "hybrid" and "finance" not in model:
            payload["output_type"] = "hybrid"
        
        try:
            # Make the request to the Voyage API
            response = requests.post(
                self.embedding_endpoint,
                headers=self.headers,
                json=payload
            )
            
            # Check for errors
            if not response.ok:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                st.error(f"Embedding API error: {response.status_code} - {error_detail}")
                return [], []
            
            # Parse the response
            data = response.json()
            
            # Extract dense embedding vector
            dense_embedding = data['data'][0]['embedding']
            
            # Extract sparse embedding (indices and values) if available
            sparse_embedding = []
            if output_type == "hybrid" and 'sparse_embedding' in data['data'][0]:
                sparse_data = data['data'][0]['sparse_embedding']
                indices = sparse_data['indices']
                values = sparse_data['values']
                sparse_embedding = list(zip(indices, values))
            
            return dense_embedding, sparse_embedding
            
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            if 'response' in locals() and hasattr(response, 'text'):
                st.error(f"API response: {response.text}")
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
        # Prepare the request payload
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n
        }
        
        try:
            # Make the request to the Voyage API
            response = requests.post(
                self.reranking_endpoint,
                headers=self.headers,
                json=payload
            )
            
            # Check for errors
            if not response.ok:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                st.error(f"Reranking API error: {response.status_code} - {error_detail}")
                return []
            
            # Parse the response
            data = response.json()
            
            return data['results']
            
        except Exception as e:
            st.error(f"Error reranking documents: {str(e)}")
            if 'response' in locals() and hasattr(response, 'text'):
                st.error(f"API response: {response.text}")
            return []

class ClaudeClient:
    """Client for Claude AI completions using direct API calls."""
    
    def __init__(self, api_key: str):
        """Initialize the Claude client with API key."""
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
                os.environ['HTTP_PROXY'] = orig_http_proxy
            if orig_https_proxy:
                os.environ['HTTPS_PROXY'] = orig_https_proxy
            if orig_http_proxy_lower:
                os.environ['http_proxy'] = orig_http_proxy_lower
            if orig_https_proxy_lower:
                os.environ['https_proxy'] = orig_https_proxy_lower
            
            # Check for errors
            if not response.ok:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                st.error(f"Claude API error: {response.status_code} - {error_detail}")
                return f"I apologize, but I encountered an error generating a response: {error_detail}"
            
            # Parse the response
            data = response.json()
            
            # Extract the response text
            return data['content'][0]['text']
            
        except Exception as e:
            st.error(f"Error generating Claude response: {str(e)}")
            if 'response' in locals() and hasattr(response, 'text'):
                st.error(f"API response: {response.text}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"

class QdrantSearch:
    """Handles search operations with Qdrant."""
    
    def __init__(self, url: str, api_key: str, collection_name: str):
        """Initialize the Qdrant search client."""
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
    
    def hybrid_search(self, 
                      dense_vector: List[float], 
                      sparse_vector: List[Tuple[int, float]], 
                      limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform search using dense vectors only if sparse vectors aren't available.
        
        Args:
            dense_vector: Dense embedding vector
            sparse_vector: Sparse embedding as list of (index, value) tuples
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            search_params = models.SearchParams(
                hnsw_ef=128,
                exact=False
            )
            
            # Prepare query vector dictionary
            query_vector = {
                "dense": dense_vector
            }
            
            # Add sparse vector if available
            if sparse_vector:
                # Convert sparse vector to the format Qdrant expects
                indices, values = zip(*sparse_vector)
                query_vector["sparse"] = models.SparseVector(
                    indices=list(indices),
                    values=list(values),
                )
            
            # Perform the search using the newer query API
            results = self.client.query(
                collection_name=self.collection_name,
                query_vector=query_vector,
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
        """Initialize all components of the RAG chatbot."""
        # Initialize Voyage AI client
        self.voyage_client = VoyageAIClient(voyage_api_key)
        
        # Initialize Claude client
        self.claude_client = ClaudeClient(claude_api_key)
        
        # Initialize Qdrant search
        self.qdrant_search = QdrantSearch(qdrant_url, qdrant_api_key, collection_name)
    
    def process_query(self, query: str, embedding_model: str = VOYAGE_EMBEDDING_MODEL, 
                     output_type: str = "dense", use_reranking: bool = True, 
                     reranker_model: str = VOYAGE_RERANKER_MODEL, top_k: int = 5) -> str:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: The user's question
            embedding_model: The model to use for generating embeddings
            output_type: Type of embedding to generate ('hybrid', 'dense', or 'sparse')
            use_reranking: Whether to use reranking
            reranker_model: The model to use for reranking
            top_k: Number of documents to retrieve
            
        Returns:
            Generated response
        """
        # Step 1: Generate embeddings for the query
        dense_embedding, sparse_embedding = self.voyage_client.create_embeddings(
            query, model=embedding_model, output_type=output_type
        )
        
        if not dense_embedding:
            return "Failed to generate embeddings for your query. Please try again."
        
        # Step 2: Perform search in Qdrant
        search_results = self.qdrant_search.hybrid_search(
            dense_vector=dense_embedding,
            sparse_vector=sparse_embedding,
            limit=top_k
        )
        
        if not search_results:
            return "No relevant documents found to answer your question."
        
        # Extract document texts from search results
        documents = [result.payload.get("text", "") for result in search_results if "text" in result.payload]
        
        if not documents:
            return "Found results, but they don't contain text fields. Check your Qdrant collection structure."
        
        # Step 3: Optionally rerank documents
        if use_reranking and len(documents) > 1:
            reranked_results = self.voyage_client.rerank_documents(
                query, documents, model=reranker_model, top_n=top_k
            )
            if reranked_results:
                # Get the reranked document texts in the new order
                documents = [documents[result['index']] for result in reranked_results]
        
        # Step 4: Generate response with Claude
        response = self.claude_client.generate_response(query, documents)
        
        return response
