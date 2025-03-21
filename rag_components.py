import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI

# Directly import both TextEmbedding and SparseTextEmbedding
try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    HAS_BM25 = True
except ImportError:
    try:
        # Try to import separately in case they're in different modules
        from fastembed import TextEmbedding
        from fastembed.sparse import SparseTextEmbedding
        HAS_BM25 = True
    except ImportError:
        # Last resort - try just the TextEmbedding
        try:
            from fastembed import TextEmbedding
            HAS_BM25 = False
        except ImportError:
            TextEmbedding = None
            HAS_BM25 = False

# Import streamlit ONLY after the above imports
import streamlit as st

# Voyage AI model configurations
VOYAGE_EMBEDDING_MODEL = "voyage-finance-2"  # Keep finance-2 as default model
VOYAGE_RERANKER_MODEL = "rerank-2"  # Latest reranker model

# OpenAI model configuration
OPENAI_MODEL = "gpt-4o"

# Vector dimensions
VECTOR_DIMENSIONS = {
    "voyage-finance-2": 1024,
    "voyage-large-3": 4096,
    "sparse": 100  # Updated for BM25 sparse embedding
}

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
    
    def create_dense_embeddings(self, text: str, model: str = VOYAGE_EMBEDDING_MODEL) -> List[float]:
        """
        Generate dense embeddings using Voyage AI.
        
        Args:
            text: The input text to embed
            model: The embedding model to use (voyage-finance-2, voyage-code-2, etc.)
            
        Returns:
            Dense embedding vector
        """
        # Prepare the request payload
        payload = {
            "model": model,
            "input": text
        }
        
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
                return []
                
            # Parse the response
            data = response.json()
            
            # Extract dense embedding vector
            dense_embedding = data['data'][0]['embedding']
            return dense_embedding
            
        except Exception as e:
            st.error(f"Error generating Voyage AI embeddings: {str(e)}")
            if 'response' in locals() and hasattr(response, 'text'):
                st.error(f"API response: {response.text}")
            return []
    
    def rerank_documents(self, query: str, documents: List[str], model: str = VOYAGE_RERANKER_MODEL, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on relevance to the query.
        
        Args:
            query: The user's question
            documents: List of documents to rerank
            model: The reranker model to use
            top_k: Number of top results to return
            
        Returns:
            List of reranked document indices with scores
        """
        # Prepare the request payload
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_k": min(top_k, len(documents))
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
            
            # Extract reranking results
            return data['results']
            
        except Exception as e:
            st.error(f"Error reranking documents: {str(e)}")
            if 'response' in locals() and hasattr(response, 'text'):
                st.error(f"API response: {response.text}")
            return []

class FastEmbedClient:
    """Client for fastembed BM25 sparse embeddings."""
    
    def __init__(self):
        """Initialize the fastembed client."""
        if HAS_BM25:
            # Use the same approach as in the ETL pipeline
            self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        else:
            self.sparse_model = None
        
    def create_sparse_embeddings(self, text: str) -> List[Tuple[int, float]]:
        """
        Generate sparse embeddings using BM25, following the exact approach used in the ETL pipeline.
        
        Args:
            text: The input text to embed
            
        Returns:
            Sparse embedding as list of (index, value) tuples
        """
        try:
            if not HAS_BM25 or not self.sparse_model:
                st.warning("BM25 encoder not available. Sparse embeddings couldn't be generated.")
                return []
                
            # Generate sparse embedding with BM25 - same as in ETL pipeline
            embeddings = list(self.sparse_model.embed([text]))
            if not embeddings:
                st.warning("Failed to generate BM25 embedding")
                return []
            
            sparse_embedding = embeddings[0]
            
            # Convert to (index, value) tuple list format for Qdrant
            indices = sparse_embedding.indices.tolist()
            values = sparse_embedding.values.tolist()
            sparse_embedding = list(zip(indices, values))
            return sparse_embedding
                
        except Exception as e:
            st.error(f"Error generating BM25 sparse embeddings: {str(e)}")
            return []

class QdrantSearch:
    """Client for Qdrant vector database search."""
    
    def __init__(self, url: str, api_key: str, collection_name: str):
        """Initialize the Qdrant client."""
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,
            prefer_grpc=False  # Force HTTP protocol
        )
        self.collection_name = collection_name
    
    def hybrid_search(self, dense_vector: List[float], sparse_vector: List[Tuple[int, float]], limit: int = 20, filter_conditions: Optional[Dict] = None) -> List[Any]:
        """
        Perform hybrid search in Qdrant using both dense and sparse vectors.
        
        Args:
            dense_vector: Dense embedding vector
            sparse_vector: Sparse embedding as list of (index, value) tuples
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions for search
            
        Returns:
            List of search results
        """
        try:
            # Prepare search parameters
            search_params = {
                "collection_name": self.collection_name,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False,
            }
            
            # Prepare query vector(s)
            if dense_vector and sparse_vector:
                # Convert sparse vector format to Qdrant's expected format
                indices = [idx for idx, _ in sparse_vector]
                values = [val for _, val in sparse_vector]
                
                # Use hybrid search with both vectors
                search_params["query_vector"] = {
                    "dense": dense_vector,
                    "sparse": models.SparseVector(
                        indices=indices,
                        values=values
                    )
                }
            elif dense_vector:
                # Use only dense vector
                search_params["query_vector"] = {
                    "dense": dense_vector
                }
            else:
                return []  # No vectors to search with
            
            # Add filter if provided
            if filter_conditions:
                search_params["query_filter"] = models.Filter(**filter_conditions)
            
            # Execute search
            results = self.client.search(**search_params)
            return results
            
        except Exception as e:
            st.error(f"Error searching Qdrant: {str(e)}")
            return []

class OpenAIClient:
    """Client for OpenAI GPT-4o completions."""
    
    def __init__(self, api_key: str, model: str = OPENAI_MODEL):
        """Initialize the OpenAI client with API key."""
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def generate_response(self, query: str, contexts: List[str], system_prompt: str = None) -> str:
        """
        Generate a response using OpenAI GPT-4o based on the query and retrieved contexts.
        
        Args:
            query: The user's question
            contexts: List of retrieved document texts
            system_prompt: Optional system prompt to customize the model's behavior
            
        Returns:
            OpenAI's response
        """
        # Combine contexts with line breaks
        combined_contexts = "\n\n".join([f"Document {i+1}:\n{context}" for i, context in enumerate(contexts)])
        
        # Create system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant that answers questions based on the provided documents. "
                "If the documents contain the information to answer the question, provide a comprehensive response based solely on that information. "
                "If the answer isn't contained in the documents, state that you don't have enough information to answer the question. "
                "Do not make up information or use your prior knowledge to answer."
            )
        
        try:
            # Make the request to the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Here are some relevant documents to help answer the question:

{combined_contexts}

Based on the above documents, please answer this question: {query}"""}
                ],
                max_tokens=2048  # Increased from 1024 to 2048
            )
            
            # Extract the response text
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating OpenAI response: {str(e)}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"

class RAGChatbot:
    """RAG Chatbot combining all components for hybrid search."""
    
    def __init__(self, qdrant_url, qdrant_api_key, collection_name, openai_api_key, voyage_api_key):
        """Initialize all components of the RAG chatbot."""
        # Initialize Voyage AI client for dense embeddings and reranking
        self.voyage_client = VoyageAIClient(voyage_api_key)
        
        # Initialize FastEmbed client for sparse embeddings
        self.fastembed_client = FastEmbedClient()
        
        # Initialize OpenAI client for response generation
        self.openai_client = OpenAIClient(openai_api_key)
        
        # Initialize Qdrant search
        self.qdrant_search = QdrantSearch(qdrant_url, qdrant_api_key, collection_name)
    
    def process_query(self, query: str) -> str:
        """
        Process a user query through the complete RAG pipeline with hybrid search.
        
        Args:
            query: The user's question
            
        Returns:
            Generated response
        """
        try:
            # Step 1: Generate embeddings for the query
            st.info("Generating embeddings...")
            
            # Generate dense embeddings with Voyage AI - use dense output type only
            dense_embedding = self.voyage_client.create_dense_embeddings(query)
            if not dense_embedding:
                return "Failed to generate dense embeddings for your query. Please try again later."
            
            # Generate sparse embeddings with BM25 separately using the ETL approach
            sparse_embedding = self.fastembed_client.create_sparse_embeddings(query)
            if not sparse_embedding and HAS_BM25:
                st.warning("Could not generate sparse embeddings. Falling back to dense search only.")
            
            # Step 2: Perform search in Qdrant
            st.info("Retrieving relevant documents...")
            try:
                search_results = self.qdrant_search.hybrid_search(
                    dense_vector=dense_embedding,
                    sparse_vector=sparse_embedding,
                    limit=20
                )
            except Exception as e:
                st.error(f"Error searching Qdrant: {str(e)}")
                # Try a fallback to dense-only search if hybrid search fails
                if "Collection doesn't exist" in str(e) or "Not found" in str(e):
                    st.warning(f"Collection might not support hybrid search. Please check your collection: {self.qdrant_search.collection_name}")
                    return f"Error: The collection '{self.qdrant_search.collection_name}' doesn't exist or doesn't support hybrid search. Please check your Qdrant settings."
                return f"Error querying database: {str(e)}"
            
            if not search_results:
                return "No relevant documents found to answer your question."
            
            # Step 3: Extract document texts from search results
            documents = [result.payload.get("chunk_text", "") for result in search_results if "chunk_text" in result.payload]
            
            # Try alternative field names if no chunk_text is found
            if not documents:
                for field in ["text", "content", "document", "context"]:
                    documents = [result.payload.get(field, "") for result in search_results if field in result.payload]
                    if documents:
                        break
            
            if not documents:
                return "Found results, but they don't contain text fields. Check your Qdrant collection structure."
            
            # Step 4: Always rerank documents
            st.info("Reranking documents...")
            reranked_results = self.voyage_client.rerank_documents(
                query, documents, top_k=20
            )
            
            if reranked_results:
                # Get the reranked document texts in the new order
                documents = [documents[result['index']] for result in reranked_results]
            else:
                st.warning("Reranking failed. Using original document order.")
            
            # Step 5: Generate response with OpenAI
            st.info("Generating response with OpenAI...")
            response = self.openai_client.generate_response(query, documents)
            return response
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
