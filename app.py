import streamlit as st
import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
from rag_components import VoyageAIClient, QdrantSearch, OpenAIClient, RAGChatbot, VOYAGE_EMBEDDING_MODEL, VOYAGE_RERANKER_MODEL, HAS_BM25

# Set page configuration FIRST - must come before any other Streamlit commands
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Now import necessary modules - AFTER page configuration
from rag_components import RAGChatbot

# Show warning about BM25 if needed - now safely AFTER set_page_config
if not HAS_BM25:
    st.warning("BM25Encoder not available in your fastembed installation. Hybrid search will not be available.")

# Voyage AI model configurations
VOYAGE_EMBEDDING_MODEL = "voyage-finance-2"  # Default model, can be overridden
VOYAGE_RERANKER_MODEL = "rerank-2"  # Latest reranker model

# OpenAI model configuration
OPENAI_MODEL = "gpt-4o"

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
        
        # Add output_type for hybrid/sparse requests
        # Only add output_type for models that support it
        if output_type in ["hybrid", "sparse"] and "finance" not in model:
            payload["output_type"] = output_type
        elif output_type in ["hybrid", "sparse"] and "finance" in model:
            # If a finance model is being used with hybrid/sparse request, log a warning
            st.warning(f"Model {model} doesn't support {output_type} embeddings. Falling back to dense only.")
        
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
            if 'sparse_embedding' in data['data'][0]:
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
    
    def rerank_documents(self, query: str, documents: List[str], model: str = VOYAGE_RERANKER_MODEL, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            model: The reranking model to use
            top_k: Number of top results to return
            
        Returns:
            List of reranked documents with scores
        """
        # Prepare the request payload - using top_k instead of top_n
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_k": top_k,
            "truncation": True
        }
        
        try:
            # Make the request to the Voyage AI API
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
                
            # Parse the response - Note: Voyage API uses 'data' field, not 'results'
            data = response.json()
            return data['data']
            
        except Exception as e:
            st.error(f"Error reranking documents: {str(e)}")
            if 'response' in locals() and hasattr(response, 'text'):
                st.error(f"API response: {response.text}")
            return []

class QdrantSearch:
    """Handles search operations with Qdrant."""
    
    def __init__(self, url: str, api_key: str, collection_name: str):
        """Initialize the Qdrant search client."""
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
    
    def hybrid_search(self, dense_vector: List[float], 
                      sparse_vector: List[Tuple[int, float]], 
                      limit: int = 5,
                      filter_params: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Qdrant's Query API with RRF fusion.
        
        Args:
            dense_vector: Dense embedding vector
            sparse_vector: Sparse embedding as list of (index, value) tuples
            limit: Maximum number of results to return
            filter_params: Optional filter parameters
            
        Returns:
            List of search results
        """
        try:
            # Convert filter if provided
            filter_obj = None
            if filter_params:
                filter_obj = models.Filter(**filter_params)
            
            # If we have both dense and sparse vectors, use the Query API with RRF fusion
            if dense_vector and sparse_vector:
                # Convert sparse vector to the format Qdrant expects
                indices, values = zip(*sparse_vector)
                
                # Execute query with RRF fusion using the client's query_points method
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=list(indices), 
                                values=list(values)
                            ),
                            using="sparse",
                            limit=limit * 2,
                            filter=filter_obj
                        ),
                        models.Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=limit * 2,
                            filter=filter_obj
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    with_payload=True
                )
                return results
                
            # If we only have dense vector, use regular search
            elif dense_vector:
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", dense_vector),
                    query_filter=filter_obj,
                    limit=limit,
                    with_payload=True
                )
                
            # If we only have sparse vector, use sparse search
            elif sparse_vector:
                indices, values = zip(*sparse_vector)
                sparse_vector_obj = models.SparseVector(
                    indices=list(indices), 
                    values=list(values)
                )
                
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("sparse", sparse_vector_obj),
                    query_filter=filter_obj,
                    limit=limit,
                    with_payload=True
                )
                
            else:
                st.error("No vectors provided for search")
                return []
                
        except Exception as e:
            st.error(f"Error searching Qdrant: {str(e)}")
            return []

class RAGChatbot:
    """RAG Chatbot combining all components."""
    
    def __init__(self, qdrant_url, qdrant_api_key, collection_name, openai_api_key, voyage_api_key, use_fastembed=False):
        """Initialize all components of the RAG chatbot."""
        # Choose between VoyageAI or FastEmbed
        if use_fastembed or not voyage_api_key:
            self.embedding_client = FastEmbedClient()
            self.use_fastembed = True
        else:
            self.voyage_client = VoyageAIClient(voyage_api_key)
            self.use_fastembed = False
        
        # Initialize OpenAI client
        self.openai_client = OpenAIClient(openai_api_key)
        
        # Initialize Qdrant search
        self.qdrant_search = QdrantSearch(qdrant_url, qdrant_api_key, collection_name)
    
    def process_query(self, query: str, embedding_model: str = VOYAGE_EMBEDDING_MODEL,
                     output_type: str = "hybrid", use_reranking: bool = True,
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
        try:
            # Step 1: Generate embeddings for the query
            st.info("Generating embeddings...")
            if self.use_fastembed:
                dense_embedding, sparse_embedding = self.embedding_client.create_embeddings(
                    query, output_type=output_type
                )
            else:
                dense_embedding, sparse_embedding = self.voyage_client.create_embeddings(
                    query, model=embedding_model, output_type=output_type
                )
            
            if not dense_embedding:
                return "Failed to generate dense embeddings for your query."
            
            # Check if sparse embeddings were generated if hybrid was requested
            if output_type in ["hybrid", "sparse"] and not sparse_embedding:
                st.warning(f"Could not generate sparse embeddings with model {embedding_model}. Falling back to dense search only.")
            
            # Step 2: Perform search in Qdrant
            st.info("Retrieving relevant documents...")
            search_results = self.qdrant_search.hybrid_search(
                dense_vector=dense_embedding,
                sparse_vector=sparse_embedding,
                limit=top_k
            )
            
            if not search_results:
                return "No relevant documents found to answer your question."
            
            # Step 3: Extract document texts from search results
            documents = [result.payload.get("chunk_text", "") for result in search_results if "chunk_text" in result.payload]
            
            # Try alternative field names if no chunk_text is found
            if not documents:
                for field in ["text", "content", "document"]:
                    documents = [result.payload.get(field, "") for result in search_results if field in result.payload]
                    if documents:
                        break
            
            if not documents:
                return "Found results, but they don't contain text fields. Check your Qdrant collection structure."
            
            # Step 4: Optionally rerank documents
            if use_reranking and len(documents) > 1:
                st.info("Reranking documents...")
                reranked_results = self.voyage_client.rerank_documents(
                    query, documents, model=reranker_model, top_k=top_k
                )
                
                if reranked_results:
                    # Get the reranked document texts in the new order
                    documents = [documents[result['index']] for result in reranked_results]
            
            # Step 5: Generate response with OpenAI
            st.info("Generating response with OpenAI...")
            response = self.openai_client.generate_response(query, documents)
            return response
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"

# Set up your Streamlit application
st.title("RAG Chatbot with Voyage AI and OpenAI")
st.write("This chatbot uses hybrid search (dense + sparse vectors) and reranking for optimal results.")

# Get API keys from environment variables or use empty strings as defaults
voyage_api_key = os.environ.get("VOYAGE_API_KEY", "")
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
qdrant_url = os.environ.get("QDRANT_URL", "")
qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
collection_name = os.environ.get("QDRANT_COLLECTION", "")

# Sidebar for API keys
with st.sidebar:
    st.header("API Configuration")
    voyage_api_key = st.text_input("Voyage AI API Key", type="password", value=voyage_api_key)
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=openai_api_key)
    qdrant_url = st.text_input("Qdrant URL", value=qdrant_url)
    qdrant_api_key = st.text_input("Qdrant API Key", type="password", value=qdrant_api_key)
    collection_name = st.text_input("Qdrant Collection Name", value=collection_name)

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chatbot_initialized" not in st.session_state:
    st.session_state.chatbot_initialized = False

# Button to initialize the chatbot
if st.button("Initialize Chatbot"):
    if all([voyage_api_key, openai_api_key, qdrant_url, qdrant_api_key, collection_name]):
        try:
            # Initialize the RAGChatbot
            st.session_state.chatbot = RAGChatbot(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                collection_name=collection_name,
                openai_api_key=openai_api_key,
                voyage_api_key=voyage_api_key
            )
            st.session_state.chatbot_initialized = True
            st.success("Chatbot initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
    else:
        st.error("Please provide all required API keys and configuration in the sidebar.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input field (only shown if chatbot is initialized)
if st.session_state.chatbot_initialized:
    prompt = st.chat_input("Ask something...")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.process_query(prompt)
                    st.markdown(response)
                    
                    # Add bot response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"Error processing query: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
